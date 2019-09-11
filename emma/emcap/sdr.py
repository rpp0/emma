import logging
import osmosdr
from gnuradio import blocks
from gnuradio import gr
from gnuradio import uhd

logger = logging.getLogger(__name__)


def set_gain(source, gain):
    source.set_gain(gain, 0)
    new_gain = source.get_gain()
    if new_gain != gain:
        raise Exception("Requested gain %.2f but set gain %.2f" % (gain, new_gain))
    return True


class SDR(gr.top_block):
    def __init__(self, hw="usrp", samp_rate=100000, freq=3.2e9, gain=0, ds_mode=False, agc=False):
        gr.enable_realtime_scheduling()
        gr.top_block.__init__(self, "SDR capture device")

        ##################################################
        # Variables
        ##################################################
        self.hw = hw
        self.samp_rate = samp_rate
        self.freq = freq
        self.gain = gain
        self.ds_mode = ds_mode
        logger.info("%s: samp_rate=%d, freq=%f, gain=%d, ds_mode=%s" % (hw, samp_rate, freq, gain, ds_mode))

        ##################################################
        # Blocks
        ##################################################
        if hw == "usrp":
            self.sdr_source = uhd.usrp_source(
               ",".join(("", "recv_frame_size=1024", "num_recv_frames=1024", "spp=1024")),
               # ",".join(("", "")),
               uhd.stream_args(
                   cpu_format="fc32",
                   channels=range(1),
               ),
            )
            self.sdr_source.set_samp_rate(samp_rate)
            self.sdr_source.set_center_freq(freq, 0)
            set_gain(self.sdr_source, gain)
            # self.sdr_source.set_min_output_buffer(16*1024*1024)  # 16 MB output buffer
            self.sdr_source.set_antenna('RX2', 0)
            self.sdr_source.set_bandwidth(samp_rate, 0)
            self.sdr_source.set_recv_timeout(0.001, True)
        else:
            if hw == "hackrf":
                rtl_string = ""
            else:
                rtl_string = "rtl=0,"
            if ds_mode:
                self.sdr_source = osmosdr.source(args="numchan=" + str(1) + " " + rtl_string + "buflen=1024,direct_samp=2")
            else:
                self.sdr_source = osmosdr.source(args="numchan=" + str(1) + " " + rtl_string + "buflen=4096")
            self.sdr_source.set_sample_rate(samp_rate)
            self.sdr_source.set_center_freq(freq, 0)
            self.sdr_source.set_freq_corr(0, 0)
            self.sdr_source.set_dc_offset_mode(0, 0)
            self.sdr_source.set_iq_balance_mode(0, 0)
            if agc:
                self.sdr_source.set_gain_mode(True, 0)
            else:
                self.sdr_source.set_gain_mode(False, 0)
                # self.sdr_source.set_if_gain(24, 0)
                # self.sdr_source.set_bb_gain(20, 0)
                set_gain(self.sdr_source, gain)
            self.sdr_source.set_antenna('', 0)
            self.sdr_source.set_bandwidth(samp_rate, 0)

        self.udp_sink = blocks.udp_sink(8, "127.0.0.1", 3884, payload_size=1472, eof=True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.sdr_source, 0), (self.udp_sink, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        if self.hw == "usrp":
            self.sdr_source.set_samp_rate(self.samp_rate)
        else:
            self.sdr_source.set_sample_rate(self.sample_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.sdr_source.set_center_freq(self.freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.sdr_source.set_gain(self.gain, 0)
