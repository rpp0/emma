#!/usr/bin/env python2

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from datetime import datetime
import time

class Tracer(gr.top_block):
    def __init__(self, samp_rate=8000000, directory="/tmp", project_name="tracer", trace_id=0):
        gr.top_block.__init__(self, "Top Block")

        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.samp_rate = samp_rate
        self.path = directory.rstrip('/') + '/' + project_name + "_" + str(trace_id) + "_" + str(datetime.now().strftime("%d%m%y_%H-%M-%S")) + "_s" + str(self.samp_rate) + ".cfile"
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.samp_rate, 0)
        self.uhd_usrp_source_0.set_gain(0, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex, self.path, False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))


def main():
    tb = Tracer(samp_rate=8000000, directory='/tmp/', project_name='test')
    tb.start()
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
