from functools import partial
import numpy as np
import io
import os
from typing import List
import shutil
import time
import warnings
import struct

from qcodes.instrument.ip import IPInstrument

import qcodes.utils.validators as vals


class AbacoDAC(IPInstrument):
    MAX_V_PP = {'AC': 1.0, 'DC': 1.7}  # Data sheet FMC144 user manual p. 14
    DAC_RESOLUTION_BITS = 16
    SAMPLES_IN_BUFFER_DIVISOR = 4
    FILENAME = "awg_file"
    FILE_LOCATION_FROM_CONTROL = "//DESKTOP-LUEGMM9/Abaco_4DSP_waveforms/"
    FILE_LOCATION_FROM_AWG = "C:\\Abaco_4DSP_waveforms\\"

    FILE_CHANNEL_MAPPING = {
        '_0': [2, 6, 10, 14, 1, 5, 9, 13],
        '_1': [4, 8, 12, 16, 3, 7, 11, 15]
    }

    NUM_CHANNELS = 8 * len(FILE_CHANNEL_MAPPING)

    max_16b2c = 32767

    def __init__(self, name, address, port,
                 initial_file='initial_file',
                 dformat='BIN',
                 **kwargs) -> None:

        # For Visa instrument: address is TCPIP0::hostname::port::SOCKET
        # self._visa_address = "TCPIP0::{:s}::{:d}::SOCKET".format(address, port)
        # super().__init__(name, self._visa_address, terminator='', **kwargs)

        super().__init__(name, address, port, **kwargs, persistent=False, terminator='')

        def return_parser(parser, inputstring):
            """
            Parses return values from instrument. Meant to be used when a query
            can return a meaningful finite number or a numeric representation
            of infinity
            Args:
                parser: Either int (for cases where system returns 1 or 0
                   to indicate state) or string (for all other responses).
                inputstring: The raw return value
            """

            inputstring = inputstring.strip()

            cmd, resp = inputstring.split(' ')

            output = parser(resp)

            return output

        self.add_parameter('max_trigger_freq', 
                           unit='Hz',
                           get_cmd=self._get_max_trigger_freq)
        self.add_parameter('data_format',
                           # ToDo: test - what happens when data format hasn't been set yet?
                           get_cmd=':SYST:WVEXTN?',
                           get_parser=partial(return_parser, str),
                           set_cmd=self._set_dformat,
                           vals=vals.Enum('TXT', 'BIN', 'txt', 'bin'))
        self.add_parameter('num_blocks',
                           get_cmd=':SYST:BLOCKS?',
                           get_parser=partial(return_parser, int))
        self.add_parameter('waveform_size',
                           get_cmd=':SYST:WVSIZE?',
                           get_parser=partial(return_parser, int))

        if not os.path.exists(self.FILE_LOCATION_FROM_CONTROL):
            raise RuntimeError(f"Can't find specified waveform file location, {self.FILE_LOCATION_FROM_CONTROL}.")

        self._file_extension = None
        self._data_object = None
        self._file_write_access = None

        self.initial_file = initial_file

        self._set_waveform_folder(self.FILE_LOCATION_FROM_AWG)
        
        # initialize if needed
        if not self._is_initialized():
            self._initialize()
        if not self._is_configured():
            self._configure_hardware()
        
        # then set file extension, set file mask to initial_file, and configure
        # (setting data format always uses self.initial file as file mask and reconfigures with the new dataformat)
        self.data_format(dformat)

        print("Abaco connected")

    def ask_raw(self, *args, **kwargs):
        response = super().ask_raw(*args, **kwargs)
        time.sleep(0.2)
        return response

    def _initialize(self):
        self.ask(':SYST:INIT')
        time.sleep(80)
        if not self._is_initialized():
            raise RuntimeError('System attempted to initialize but was unsuccessful.')

    def _configure_hardware(self):
        if not self._is_initialized:
            self._initialize()
        self.ask(':SYST:CONF')
        time.sleep(60)
        if not self._is_configured():
            raise RuntimeError('System attempted to configure hardware but was unsuccessful.')

    def _set_file_mask(self, filename):
        file_mask = f"{filename}_"
        self.ask(f":SYST:FMSK {file_mask}")

    def _set_waveform_folder(self, folder):
        self.ask(f':SYST:WVFLD {folder}')

    def _set_dformat(self, dformat):
        if dformat.upper() == 'TXT':
            self._file_extension = 'txt'
            self._data_object = io.StringIO
            self._file_write_access = 'w'
        elif dformat.upper() == 'BIN':
            self._file_extension = 'bin'
            self._data_object = io.BytesIO
            self._file_write_access = 'wb'
        self.ask(f':SYST:WVEXTN {dformat.upper()}')

    def _load_waveform_to_fpga(self):
        # System must be initialized and configured, and not currently outputting
        if not self._is_configured():
            raise RuntimeError("System is not configured. Cannot upload waveform to fpga.")
        if self._output_is_enabled():
            self._disable_output()
        self.ask(':SYST:LDWVF')

    def _enable_output(self):
        if not self._wf_uploaded_to_fpga():
            raise RuntimeError('Waveform not uploaded, cannot enable output')
        self.ask(':SYST:ENBL')

    def _disable_output(self):
        if not self._output_is_enabled():
            raise RuntimeError("Waveform output not enabled, cannot disable output")
        self.ask(':SYST:DSBL')

    def _get_max_trigger_freq(self):
        num_elements = self.num_blocks()
        total_samples = self.waveform_size() / 2
        # ToDo: if Ruben updates it to return size in number of samples, remove the factor of 2!!!

        samples_per_waveform = total_samples/num_elements
        waveform_size_bytes = samples_per_waveform * 2
        max_data_rate_per_channel = (12.16/8) * 1e9  # 4DSP verification records pg 16

        return int(max_data_rate_per_channel/waveform_size_bytes)

    def load_waveform_from_file(self, new_waveform_file=None):
        # update file name if using new file
        if new_waveform_file is not None:
            self._set_file_mask(new_waveform_file)
        self._load_waveform_to_fpga()

    def _get_waveform_shape_for_current_file(self):
        cmd, num_blocks, size = self.ask(':SYST:FILE?').strip().split(' ')
        return [int(num_blocks), int(size)]  # (number of elements, total waveform size per channel)

    def run(self):
        # if output has previously been stopped, the current file must be reuploaded to the fpga
        if not self._wf_uploaded_to_fpga():
            self._load_waveform_to_fpga()
        self._enable_output()

    def stop(self):
        self._disable_output()
 
    ###########################
    # System status functions #
    ###########################

    def _get_status(self, cmd):
        status = self.ask(cmd)
        if status is None:
            raise RuntimeError(f'No response from abaco when asking {cmd}')
        return int(status[-1])

    def _is_initialized(self):
        return self._get_status(':SYST:INIT?')

    def _is_configured(self):
        return self._get_status(':SYST:CONF?')

    def _wf_uploaded_to_fpga(self):
        return self._get_status(':SYST:LDWVF?')

    def _output_is_enabled(self):
        return self._get_status(':SYST:ENBL?')

    def _output_is_disabled(self):
        return self._get_status(':SYST:DSBL?')

    ######################
    # AWG file functions #
    ######################

    def make_and_send_awg_file(self, seq: List[np.ndarray], filename=None):
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
            filename: mask for the file name (files saved will then be filename_0.bin, filename_1.bin, etc)
        """
        start = time.clock()
        print('Starting to make awg file')

        used_channels = [ch for ch in seq[0]['data'].keys()]
        for ch in used_channels:
            if ch not in range(1, self.NUM_CHANNELS+1):
                warnings.warn(f"Unknown channel specified: {ch}. AWG has channels 1-{self.NUM_CHANNELS}. "
                              f"Data for {ch} will not be uploaded.")

        # get element size (size of longest channel array)
        # assume the longest array for element 0 is also the longest output array for the entire sequence
        block_size = max([len(a) for a in seq[0]['data'].values()])

        # create output dictionary containing list of output data for all channels, including padding on each element
        output_dict = {ch: [] for ch in range(1, self.NUM_CHANNELS + 1)}
        for element in seq:
            for ch in output_dict:
                for rep in range(element['sequencing']['nrep']):
                    if ch in element['data']:
                        a = self.forged_seq_array_to_16b2c(element['data'][ch])
                        output_dict[ch].append(a)
                    else:
                        output_dict[ch].append(np.zeros(block_size))


        # get number of blocks (elements), padded_block_size and total_num_samples
        n_blocks = len(output_dict[1])
        d = self.SAMPLES_IN_BUFFER_DIVISOR
        padded_block_size = -(-block_size // d) * d
        total_num_samples = padded_block_size * n_blocks

        header = self._make_file_header(n_blocks, total_num_samples)

        new_time = time.clock()
        data = self._make_file_data(n_blocks, padded_block_size, output_dict)
        print(f"Created file data in {time.clock()-new_time} seconds")
        new_time = time.clock()
        self._create_files(header, data, filename)
        print(f"Information saved to file in {time.clock()-new_time} seconds")

        end = time.clock()
        print(f"Completed making and saving file in {(end-start)} seconds")

    def _make_file_header(self, n_blocks, total_num_samples, channels_per_file=8):
        """args: number of elements, total number of samples
        returns: IO for file header, as either a StringIO or BytesIO, depending on current data_format
        """

        header = self._data_object()
        contents = [n_blocks]
        for i in range(channels_per_file):
            contents.append(total_num_samples)

        # binary format is 9 lines of 4 byte unsigned integers ('9I')
        self.write_sample(header, contents, self.data_format(), binary_format='9I')
        
        return header

    def _make_file_data(self, n_blocks, padded_block_size, output_dict):

        # ToDo: it feels like there must be a better way to organize the output_dict so that this is better
        data = {}

        for file_i, channel_list in self.FILE_CHANNEL_MAPPING.items():
            file_output_array = [output_dict[ch] for ch in channel_list]

            output = self._data_object()

            all_samples = []

            for i_block in range(n_blocks):
                for i_sample in range(padded_block_size): 
                    for i_channel in range(len(channel_list)):
                        a = file_output_array[i_channel][i_block]
                        try:
                            current_sample = int(a[i_sample])
                        except IndexError:
                            current_sample = int(0)
                        all_samples.append(current_sample)

            # binary format is len(all_samples) lines of 2 byte signed integers ('h')
            binary_format = str(len(all_samples)) + 'h'
            self.write_sample(output, all_samples, self.data_format(), binary_format)
            data[file_i] = output

        return data

    def _create_files(self, header, data, filename=None):
        if filename is None:
            filename = self.FILENAME
        
        filepath = self.FILE_LOCATION_FROM_CONTROL + filename + '{}.{}'

        file_access = self._file_write_access
        file_type = self._file_extension

        # write files to disk
        for i in data:
            contents = self._data_object()
            contents.write(header.getvalue())
            contents.write(data[i].getvalue())

            with open(filepath.format(i, file_type), file_access) as fd:
                contents.seek(0)
                shutil.copyfileobj(contents, fd, length=16*1024*1024*8)


    @staticmethod
    def write_sample(stream, contents, dformat, binary_format):
        if dformat == 'TXT':
            contents = [str(x)+'\n' for x in contents]
            stream.writelines(contents)
        elif dformat == 'BIN':
            stream.write(struct.pack(binary_format, *contents))
           
    @classmethod
    def forged_seq_array_to_16b2c(cls, array):
        """Takes an array with values between -1 and 1, where 1 specifies max voltage and -1 min voltage.
        Converts the array into twos-complement data, as required by the AWG."""
        return (array * cls.max_16b2c).astype(int)
