from RsSmw import *
import time


def down_signal(brand, x, fc, fs, power=-30, IP="192.168.0.25", file_name="Waveform_lz.wv", logger=None):
    x = x.copy()
    if brand.lower() == "rohde-schwarz":
        # MATLAB移植时碰到很多官方的文件，难以移植，故直接采用官方Python包
        pc_wv_file = "./arbFile.wv"
        instr_wv_file_out = "/var/user/" + file_name
        smw = RsSmw("TCPIP::" + IP + "::HISLIP")
        if logger:
            logger.debug("Sucessfully Connected to signal generator!")
        # RsSmw.assert_minimum_version('5.0.44')
        # print(smw.utilities.idn_string)
        smw.utilities.reset()

        # I-component an Q-component data
        i_data = [data.real for data in x]
        q_data = [data.imag for data in x]
        smw.arb_files.create_waveform_file_from_samples(
            i_data,
            q_data,
            pc_wv_file,
            clock_freq=fs,
            auto_scale=True,
            comment="Created from I/Q vectors",
        )
        smw.arb_files.send_waveform_file_to_instrument(pc_wv_file, instr_wv_file_out)

        smw.source.bb.arbitrary.waveform.set_select(instr_wv_file_out)
        smw.source.frequency.set_frequency(fc)
        smw.source.power.level.immediate.set_amplitude(power)
        smw.source.bb.arbitrary.set_state(True)
        smw.output.state.set_value(True)

        smw.close()
        if logger:
            logger.debug("SMW signal transmission finished!")
        time.sleep(2)
