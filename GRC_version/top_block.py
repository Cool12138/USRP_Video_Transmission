#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Top Block
# GNU Radio version: 3.7.13.5
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import blocks
from gnuradio import digital
from gnuradio import dtv
from gnuradio import eng_notation
from gnuradio import fft
from gnuradio import filter
from gnuradio import gr
from gnuradio import uhd
from gnuradio import wxgui
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.wxgui import fftsink2
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import pmt
import time
import wx


class top_block(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Top Block")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.wxgui_fftsink2_0_0 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=1200000000,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=10000000,
        	fft_size=1024,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot_RX',
        	peak_hold=False,
        	win=window.rectangular,
        )
        self.Add(self.wxgui_fftsink2_0_0.win)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.GetWin(),
        	baseband_freq=1200000000,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=10000000,
        	fft_size=2048,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot_TX',
        	peak_hold=False,
        	win=window.blackmanharris,
        )
        self.Add(self.wxgui_fftsink2_0.win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(('', "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_samp_rate(2000000)
        self.uhd_usrp_source_0.set_center_freq(2400000000, 0)
        self.uhd_usrp_source_0.set_gain(19, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_auto_dc_offset(True, 0)
        self.uhd_usrp_source_0.set_auto_iq_balance(True, 0)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
        	",".join(('', "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_sink_0.set_samp_rate(2e6)
        self.uhd_usrp_sink_0.set_center_freq(24e8, 0)
        self.uhd_usrp_sink_0.set_gain(15, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccc(
                interpolation=64,
                decimation=70,
                taps=None,
                fractional_bw=None,
        )
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=70,
                decimation=64,
                taps=None,
                fractional_bw=None,
        )
        self.fft_vxx_0_0 = fft.fft_vcc(2048, True, (window.rectangular(2048)), True, 1)
        self.fft_vxx_0 = fft.fft_vcc(2048, False, (window.rectangular(2048)), True, 10)
        self.dtv_dvbt_viterbi_decoder_0 = dtv.dvbt_viterbi_decoder(dtv.MOD_16QAM, dtv.NH, dtv.C1_2, 768)
        self.dtv_dvbt_symbol_inner_interleaver_0_0 = dtv.dvbt_symbol_inner_interleaver(1512, dtv.T2k, 1)
        self.dtv_dvbt_symbol_inner_interleaver_0 = dtv.dvbt_symbol_inner_interleaver(1512, dtv.T2k, 1)
        self.dtv_dvbt_reference_signals_0 = dtv.dvbt_reference_signals(gr.sizeof_gr_complex, 1512, 2048, dtv.MOD_16QAM, dtv.NH, dtv.C1_2, dtv.C1_2, dtv.GI_1_32, dtv.T2k, 0, 0)
        self.dtv_dvbt_reed_solomon_enc_0 = dtv.dvbt_reed_solomon_enc(2, 8, 285, 255, 239, 8, 51, 8)
        self.dtv_dvbt_reed_solomon_dec_0 = dtv.dvbt_reed_solomon_dec(2, 8, 0x11d, 255, 239, 8, 51, 8)
        self.dtv_dvbt_ofdm_sym_acquisition_0 = dtv.dvbt_ofdm_sym_acquisition(1, 2048, 1705, 64, 30)
        self.dtv_dvbt_map_0 = dtv.dvbt_map(1512, dtv.MOD_16QAM, dtv.NH, dtv.T2k, 1)
        self.dtv_dvbt_inner_coder_0 = dtv.dvbt_inner_coder(1, 1512, dtv.MOD_16QAM, dtv.NH, dtv.C1_2)
        self.dtv_dvbt_energy_dispersal_0 = dtv.dvbt_energy_dispersal(1)
        self.dtv_dvbt_energy_descramble_0 = dtv.dvbt_energy_descramble(8)
        self.dtv_dvbt_demod_reference_signals_0 = dtv.dvbt_demod_reference_signals(gr.sizeof_gr_complex, 2048, 1512, dtv.MOD_16QAM, dtv.NH, dtv.C1_2, dtv.C1_2, dtv.GI_1_32, dtv.T2k, 0, 0)
        self.dtv_dvbt_demap_0 = dtv.dvbt_demap(1512, dtv.MOD_16QAM, dtv.NH, dtv.T2k, 1)
        self.dtv_dvbt_convolutional_interleaver_0 = dtv.dvbt_convolutional_interleaver(136, 12, 17)
        self.dtv_dvbt_convolutional_deinterleaver_0 = dtv.dvbt_convolutional_deinterleaver(136, 12, 17)
        self.dtv_dvbt_bit_inner_interleaver_0 = dtv.dvbt_bit_inner_interleaver(1512, dtv.MOD_16QAM, dtv.NH, dtv.T2k)
        self.dtv_dvbt_bit_inner_deinterleaver_0 = dtv.dvbt_bit_inner_deinterleaver(1512, dtv.MOD_16QAM, dtv.NH, dtv.T2k)
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(2048, 2048+64, 0, '')
        self.blocks_vector_to_stream_0_0 = blocks.vector_to_stream(gr.sizeof_char*1, 1512)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_vcc((0.00220971, ))
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vcc((0.00220971, ))
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/baker/GRC_test/test_11.4M.ts', False)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_char*1, '/home/baker/Desktop/rec_video.ts', True)
        self.blocks_file_sink_0_0.set_unbuffered(False)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.dtv_dvbt_energy_dispersal_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.dtv_dvbt_ofdm_sym_acquisition_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.wxgui_fftsink2_0_0, 0))
        self.connect((self.blocks_vector_to_stream_0_0, 0), (self.dtv_dvbt_viterbi_decoder_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.dtv_dvbt_bit_inner_deinterleaver_0, 0), (self.blocks_vector_to_stream_0_0, 0))
        self.connect((self.dtv_dvbt_bit_inner_interleaver_0, 0), (self.dtv_dvbt_symbol_inner_interleaver_0, 0))
        self.connect((self.dtv_dvbt_convolutional_deinterleaver_0, 0), (self.dtv_dvbt_reed_solomon_dec_0, 0))
        self.connect((self.dtv_dvbt_convolutional_interleaver_0, 0), (self.dtv_dvbt_inner_coder_0, 0))
        self.connect((self.dtv_dvbt_demap_0, 0), (self.dtv_dvbt_symbol_inner_interleaver_0_0, 0))
        self.connect((self.dtv_dvbt_demod_reference_signals_0, 0), (self.dtv_dvbt_demap_0, 0))
        self.connect((self.dtv_dvbt_energy_descramble_0, 0), (self.blocks_file_sink_0_0, 0))
        self.connect((self.dtv_dvbt_energy_dispersal_0, 0), (self.dtv_dvbt_reed_solomon_enc_0, 0))
        self.connect((self.dtv_dvbt_inner_coder_0, 0), (self.dtv_dvbt_bit_inner_interleaver_0, 0))
        self.connect((self.dtv_dvbt_map_0, 0), (self.dtv_dvbt_reference_signals_0, 0))
        self.connect((self.dtv_dvbt_ofdm_sym_acquisition_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.dtv_dvbt_reed_solomon_dec_0, 0), (self.dtv_dvbt_energy_descramble_0, 0))
        self.connect((self.dtv_dvbt_reed_solomon_enc_0, 0), (self.dtv_dvbt_convolutional_interleaver_0, 0))
        self.connect((self.dtv_dvbt_reference_signals_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.dtv_dvbt_symbol_inner_interleaver_0, 0), (self.dtv_dvbt_map_0, 0))
        self.connect((self.dtv_dvbt_symbol_inner_interleaver_0_0, 0), (self.dtv_dvbt_bit_inner_deinterleaver_0, 0))
        self.connect((self.dtv_dvbt_viterbi_decoder_0, 0), (self.dtv_dvbt_convolutional_deinterleaver_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.dtv_dvbt_demod_reference_signals_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.wxgui_fftsink2_0, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.rational_resampler_xxx_0_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate


def main(top_block_cls=top_block, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
