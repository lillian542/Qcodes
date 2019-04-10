from contextlib import contextmanager
import numpy as np
import io
from typing import List
from math import ceil
import shutil
import time

import time

from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.ip import IPInstrument


class AbacoDAC(IPInstrument):
    V_PP_DC = 1.7  # Data sheet FMC144 user manual p. 14
    V_PP_AC = 1.0  # Data sheet FMC144 user manual p. 14
    DAC_RESOLUTION_BITS = 16
    SAMPLES_IN_BUFFER_DIVISOR = 4
    FILENAME = "test_{}.{}"

    FILE_CHANNEL_POSITION = {
        '_0': [1, 2, 3, 4, 5, 6, 7, 8],
        '_1': [9, 10, 11, 12, 13, 14, 15, 16]
    }

    NUM_CHANNELS = 8 * len(FILE_CHANNEL_POSITION)

    max_16b2c = 32767

    def __init__(self, name, address, port, *args, **kwargs) -> None:
        # address is TCPIP0::hostname::port::SOCKET
        # self._visa_address = "TCPIP0::{:s}::{:d}::SOCKET".format(address, port)
        # super().__init__(name, self._visa_address, terminator='', **kwargs)
        super().__init__(name, address, port, *args, **kwargs, persistent=False, terminator='')
        # with self.temporary_timeout(11):
        #     print("asked returned {}".format(self.ask("init_state\n")))
            #print("asked returned {}".format(self.ask("init_state\n")))
        # # cls.ask("init_state")
        # # time.sleep(1)
        # # cls.ask("config_state")
        # # glWaveFileMask=test_
        # pass
        self.add_function('_initialize', call_cmd='init_state')
        # ToDo: (Ruben) all of these things should throw an error if they don't work because the system is in the wrong state (currently error only prints to command terminal)
        self.add_function('_hardware_configure', call_cmd='config_state')
        self.add_function('_load_waveform_to_fpga', call_cmd='load_waveform_state')

        self.output_enabled = False

    @contextmanager
    def temporary_timeout(self, timeout):
        old_timeout = self._timeout
        self.set_timeout(timeout)
        yield
        self.set_timeout(old_timeout)

    def _enable_output(self):
        self.ask('enable_offload_state')
        self.output_enabled = True

    def _disable_output(self):
        self.ask('disable_offload_state')
        self.output_enabled = False

    def _is_new_waveform_shape(self, new_waveform):

        self.ask('current_waveform_what_is_your_shape')
        #ToDo: talk to Ruben about how to retrieve current waveform shape info, make this function work

        self.get_waveform_shape(new_waveform)
        # ToDo: create this function (what format does the new waveform have? is it a forged sequence? a summary?)

        # ToDo: compare current and new waveforms, return False if they have the same shape, else True

        return True

    def _specify_file_for_upload(self, file):
        #ToDo: talk to Ruben about how to specify which file to upload without accessing the GUI
        pass

    def upload_to_fpga(self, file=None):
        # ToDo: select file to upload
        # reuploads last used waveform if no file is specified
        start = time.clock()
        new_shape = self._is_new_waveform_shape(file)

        # output must be disabled before any other change of intrument state can occur
        self._disable_output()

        self._specify_file_for_upload(file)

        # must reinitialize and reconfigure hardware if the new waveform does not have the same basic shape (number of points, number of blocks)
        if new_shape:
            self._initialize()
            time.sleep(80)
            self._hardware_configure()
            time.sleep(60)

        self._load_waveform_to_fpga()
        self._enable_output()
        print(f'Upload to FPGA completed in {time.clock()-start}')

    def run(self):
        if not self.output_enabled:
        # ToDo: only reload output if it hasn't been enabled and disabled
            self._load_waveform_to_fpga()
            self._enable_output()

        # ToDo: then start triggers? Or does run not make sense when the AWG only operates in external trigger mode?

    def stop(self):
        pass
        # ToDo: should this disable output or just stop the triggers? If it disables output, you need to go back to upload before you can run again.

    ######################
    # AWG file functions #
    ######################

    @classmethod
    def _create_files(cls, header, data, dformat):
        if dformat == 1:
            file_access = 'w'
            file_type = 'txt'
            content_type = io.StringIO
        else:
            file_access = 'wb'
            file_type = 'bin'
            # ToDo: is this correct?
            content_type = io.BytesIO

        # write files to disk
        for i in data:
            print(i)
            contents = content_type()
            contents.write(header.getvalue())
            contents.write(data[i].getvalue())

            with open(cls.FILENAME.format(i, file_type), file_access) as fd:
                contents.seek(0)
                shutil.copyfileobj(contents, fd)

    @classmethod
    def make_and_save_awg_file_locally(cls, seq: List[np.ndarray], dformat: int) -> str:
        """
        This function produces a text data file for the abaco DAC that
        specifies the waveforms. Samples are represented by integer values.
        The file has the following structure:
        (lines starting with '#' are not part of the file)
        #--Header--------------------------------------------------------------
        <number of blocks>
        <total number of samples channel 1>
        <total number of samples channel 2>
        ...
        <total number of samples channel 8>
        #--Block 1-------------------------------------------------------------
        <sample 1, channel 1>
        <s1-c2>
        <s1-c3>
        ...
        <s1-c8>
        <s2-c1>
        <s2-c2>
        ....
        <sN-c8>
        #--Block 2-------------------------------------------------------------
        <s1-c8>
        ....
        <sN-c8>
        #--Block 3-------------------------------------------------------------
        ...
        Please note that all blocks have to have the same length
        Args:
            seq: The forged sequence
            dformat: 1 for text file format, 2 for binary
        """

        """
        Assumptions:
            1. All elements have the same channels outputting in them, so the channels in seq[0] are the same as in
               every other element.
            2. The length (in samples) of the longest channel output array for element 0 is also the length of the 
               longest output array for the entire sequence.
            3. The total number of samples (i.e. block or element size) is the same for all channels

        """

        # ToDo: currently the sequences are output with both channels and markers.
        #       The marker names in the forged sequence will need to be converted to channel numbers to upload here

        # get element size (size of longest channel output array)
        block_size = max([len(a) for a in seq[0]['data'].values()])  # Assumption 2

        # create output dictionary containing list of output data for all channels, including padding on each element
        output_dict = {ch: [] for ch in range(1, cls.NUM_CHANNELS + 1)}
        for element in seq:
            for ch in output_dict:
                for rep in range(element['sequencing']['nrep']):
                    if ch in element['data']:
                        # ToDo: convert output dict values into twos complement data here
                        a = element['data'][ch]
                        output_dict[ch].append(a)
                    else:
                        output_dict[ch].append(np.zeros(block_size))

        # get number of blocks (elements), padded_block_size and total_num_samples
        n_blocks = len(output_dict[1])
        d = cls.SAMPLES_IN_BUFFER_DIVISOR
        padded_block_size = -(-block_size // d) * d
        total_num_samples = padded_block_size * n_blocks

        header = cls._make_file_header(n_blocks, total_num_samples, dformat)

        data = cls._make_file_data(n_blocks, padded_block_size, output_dict, dformat)

        cls._create_files(header, data, dformat)

        return header, data

    @classmethod
    def _make_file_header(cls, n_blocks, total_num_samples, dformat: int, channels_per_file=8):
        """args: forged sequence
        returns: IO for file header, in either string (dformat=1) or binary (dformat=2) format.
        """
        # ToDo: does this really need to be its own function?
        if dformat == 1:
            header = io.StringIO()
        elif dformat == 2:
            header = io.BytesIO()
        else:
            raise RuntimeError(f"Variable dformat must be 1 (for txt file) or 2 (for bin file). Received {dformat}.")

        cls.write_sample(header, n_blocks, dformat)
        for i in range(channels_per_file):
            cls.write_sample(header, total_num_samples, dformat)

        return header

    @classmethod
    def _voltage_to_int(cls, v):
        return int(round(v / cls.V_PP_DC * 2 ** (cls.DAC_RESOLUTION_BITS - 1)))

    @classmethod
    def _make_file_data(cls, n_blocks, padded_block_size, output_dict, dformat):

        # ToDo: it feels like there must be a better way to organize the output_dict so that this is better

        data = {}

        for file_i, channel_list in cls.FILE_CHANNEL_POSITION.items():
            file_output_array = [output_dict[ch] for ch in channel_list]

            if dformat == 1:
                output = io.StringIO()
            elif dformat == 2:
                output = io.BytesIO()
            else:
                raise RuntimeError(f"Variable dformat must be 1 (txt file) or 2 (bin file). Received {dformat}.")
            # ToDo: I think just check that dformat has a correct value with a validator, instead of this over and over

            for i_block in range(n_blocks):
                for i_sample in range(padded_block_size):  # Assumption 3
                    for i_channel in range(len(channel_list)):
                        a = file_output_array[i_channel][i_block]
                        try:
                            current_sample = int(a[i_sample])
                        except IndexError:
                            current_sample = int(0)
                        cls.write_sample(output, current_sample, dformat)

            data[file_i] = output

        return data

    @staticmethod
    def write_sample(stream, sample, dformat):
        if dformat == 1:
            print('{}'.format(sample), file=stream)
        elif dformat == 2:
            stream.write(sample.to_bytes(4, byteorder='big', signed=True))
