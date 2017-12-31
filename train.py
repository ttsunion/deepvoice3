# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
fangshuming519@gmail.com. 
https://www.github.com/FonzieTree/deepvoice3
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from networks import encoder, decoder, converter
import tensorflow as tf
from utils import *
import time

class Graph:
    def __init__(self, training=True):
        # Load vocabulary
        self.char2idx, self.idx2char = load_vocab()

        # Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Data Feeding
            ## x: Text. (N, Tx), int32
            ## y1: Melspectrogram. (N, Ty, n_mels) float32
            ## y2: Dones. (N, Ty) int32
            ## z: Magnitude. (N, Ty, n_fft//2+1) float32
            if training:
                self.x, self.y1, self.y2, self.z = get_batch()
                self.prev_max_attentions_li = tf.ones(shape=(hp.dec_layers, hp.batch_size), dtype=tf.int32)
            else: # Inference
                self.x = tf.placeholder(tf.int32, shape=(hp.batch_size, hp.Tx))
                self.y1 = tf.placeholder(tf.float32, shape=(hp.batch_size, hp.Ty//hp.r, hp.n_mels*hp.r))
                self.prev_max_attentions_li = tf.placeholder(tf.int32, shape=(hp.dec_layers, hp.batch_size,))

            # Get decoder inputs: feed last frames only (N, Ty, n_mels)
            self.decoder_input = tf.concat((tf.zeros_like(self.y1[:, :1, -hp.n_mels:]), self.y1[:, :-1, -hp.n_mels:]), 1)

            # Networks
            with tf.variable_scope("encoder"):
                self.keys, self.vals = encoder(self.x, training=training) # (N, Tx, e)

            with tf.variable_scope("decoder"):
                # mel_logits: (N, Ty, n_mels)
                # done_output: (N, Ty, 2),
                # decoder_output: (N, Ty, e)
                # alignments_li: dec_layers*(Tx, Ty)
                # max_attentions_li: dec_layers*(N, T_y)
                self.mel_logits, self.done_output, self.decoder_output, self.alignments_li, self.max_attentions_li = decoder(self.decoder_input,
                             self.keys,
                             self.vals,
                             self.prev_max_attentions_li,
                             training=training)
                self.mel_output = tf.nn.sigmoid(self.mel_logits)

            with tf.variable_scope("converter"):
                # Restore shape
                self.converter_input = tf.reshape(self.decoder_output, (-1, hp.Ty, hp.embed_size))
                self.converter_input = fc_block(self.converter_input,
                                                hp.converter_channels,
                                                activation_fn=tf.nn.relu,
                                                training=training) # (N, Ty, v)

                # Converter
                self.mag_logits = converter(self.converter_input, training=training) # (N, Ty, 1+n_fft//2)
                self.mag_output = tf.nn.sigmoid(self.mag_logits)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if training:
                # Loss
                self.loss_mels = tf.reduce_mean(tf.abs(self.mel_output - self.y1))
                self.loss_dones = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.done_output, labels=self.y2))
                self.loss_mags = tf.reduce_mean(tf.abs(self.mag_output - self.z))
                self.loss = self.loss_mels + self.loss_dones + self.loss_mags

                # Training Scheme
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                ## gradient clipping
                self.gvs = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.gvs:
                    grad = tf.clip_by_value(grad, -1. * hp.max_grad_val, hp.max_grad_val)
                    grad = tf.clip_by_norm(grad, hp.max_grad_norm)
                    self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)
                   
                # Summary
                tf.summary.scalar('Train_Loss/LOSS', self.loss)
                tf.summary.scalar('Train_Loss/mels', self.loss_mels)
                tf.summary.scalar('Train_Loss/dones', self.loss_dones)
                tf.summary.scalar('Train_Loss/mags', self.loss_mags)

                self.merged = tf.summary.merge_all()

if __name__ == '__main__':
    start_time = time.time()
    g = Graph(); print("Training Graph loaded")
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0)
        with sv.managed_session() as sess:
            # plot initial alignments
            alignments_li = sess.run(g.alignments_li)
            plot_alignment(alignments_li, 0)  # (Tx, Ty)

            while 1:
                if sv.should_stop(): break
                for i in range(10000):
                    gs, _ = sess.run([g.global_step, g.train_op])
                    print('Step: ', gs, '   Mels loss: ', sess.run(g.loss_mels), '    Dones loss: ', sess.run(g.loss_dones), '    Mags loss: ', sess.run(g.loss_mags))
                    # Write checkpoint files at every epoch
                    if gs % 1000 == 0:
                        sv.saver.save(sess, hp.logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                        # plot alignments
                        alignments_li = sess.run(g.alignments_li)
                        plot_alignment(alignments_li, str(gs // 1000).zfill(3) + "k")  # (Tx, Ty)


    print("Done")
