from contextlib import contextmanager
import numpy as np
import io
import os
from typing import List
from math import ceil
import shutil
import sys
import time
import warnings

from qcodes.instrument.visa import VisaInstrument
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

    STATES = {0: 'not_initialized',
              1: 'initialized',
              2: 'hardware_configured',
              3: 'wavefrom_uploaded',
              4: 'output_enabled',
              5: 'output_disabled'}
    # ToDo: this will be redundant once get_state function is implemented

    max_16b2c = 32767

    def __init__(self, name, address, port,
                 initial_file='initial_file',
                 dformat='BIN',
                 new_initialization=True,
                 **kwargs) -> None:
        """            dformat: 1 for text file format, 2 for binary """
        # address is TCPIP0::hostname::port::SOCKET
        # self._visa_address = "TCPIP0::{:s}::{:d}::SOCKET".format(address, port)
        # super().__init__(name, self._visa_address, terminator='', **kwargs)
        super().__init__(name, address, port, **kwargs, persistent=False, terminator='')
        # with self.temporary_timeout(11):
        #     print("asked returned {}".format(self.ask("init_state\n")))
        #     print("asked returned {}".format(self.ask("init_state\n")))
        # # cls.ask("init_state")
        # # time.sleep(1)
        # # cls.ask("config_state")
        # # glWaveFileMask=test_
        # pass

        self.add_parameter('max_trigger_freq', 
                           unit='Hz',
                           get_cmd=self._get_max_trigger_freq)
        self.add_parameter('data_format', 
                           get_cmd=self._get_dformat,
                           set_cmd=self._set_dformat,
                           vals = vals.Enum('TXT', 'BIN'))

        if not os.path.exists(self.FILE_LOCATION_FROM_CONTROL):
            raise RuntimeError(f"Can't find specified waveform file location, {self.FILE_LOCATION_FROM_CONTROL}.")

        self.ask(f'glWaveFileFolder={self.FILE_LOCATION_FROM_AWG}')

        self._file_extension = None
        self._data_object = None
        self._file_write_access = None

        # ToDo: decide on shape for initial file
        self.initial_file = initial_file
        self._state = 0

        if new_initialization:
            # ToDo: decide whether to initialize/configure based on get_state function
            self._initialize()
            self.data_format(dformat)
        else:
            # ToDo: this should only happen if the current dformat is not the same as the specified one
            self.data_format(dformat)
            self._state = 2
        
        self._shape = self.get_waveform_shape(initial_file)

        print("Abaco connected")

    @contextmanager
    def temporary_timeout(self, timeout):
        old_timeout = self._timeout
        self.set_timeout(timeout)
        yield
        self.set_timeout(old_timeout)

    def _initialize(self):
        self.ask('init_state')
        time.sleep(80)  # ToDo: do this with temporary timeout instead? How do timeouts work for instruments?
        self._state = 1

    def _configure_hardware(self):
        self.ask('config_state')
        time.sleep(60)
        self._state = 2

    def _set_file_mask(self, filename):
        file_mask = f"{filename}_"
        self.ask(f"glWaveFileMask={file_mask}")

    def _set_dformat(self, dformat):
        if dformat == 'TXT':
            self._file_extension = 'txt'
            self._data_object = io.StringIO
            self._file_write_access = 'w'
        else:
            self._file_extension = 'bin'
            self._data_object = io.BytesIO
            self._file_write_access = 'wb'

        self.ask(f'glWaveFileExtension=.{self._file_extension}')
        self._set_file_mask(self.initial_file)
        self._configure_hardware()

    def _get_dformat(self):
        # ToDo: can this be from get_status, instead of just saved as an attribute of the instrument?
        if isinstance(self._file_extension, str):
            return self._file_extension.strip('.').upper()
        else:
            raise RuntimeError('data format has not been set')

    def _load_waveform_to_fpga(self):
        # ToDo: Add error message if in wrong state
        self.ask('load_waveform_state')
        self._state = 3

    def _enable_output(self):
        if self._state != 3:
            raise RuntimeError('Waveform not uploaded, cannot enable output')
            # ToDo: change this to re-upload current file?

        self.ask('enable_offload_state')

        self._state = 4

    def _disable_output(self):
        if self._state != 4:
            raise RuntimeError("Waveform output not enabled, cannot disable output")

        self.ask('disable_offload_state')

        self._state = 5

    def _get_max_trigger_freq(self):
        # ToDo: update to access get_state function?
        num_elements = self._shape[0]
        total_samples = self._shape[1]

        samples_per_waveform = total_samples/num_elements
        waveform_size_bytes = samples_per_waveform * 2
        max_data_rate_per_channel = (12.16/8) * 1e9  # 4DSP verification records pg 16

        return int(max_data_rate_per_channel/waveform_size_bytes)

    def _is_new_waveform_shape(self, new_waveform):

        # ToDo: implement once get_state function is available, until then always reconfigure

        # if new_waveform is not None:
        #     new_shape = self.get_waveform_shape(new_waveform)

        #     if new_shape != self._shape:
        #         return True

        return False

    def get_waveform_shape(self, filename):
        # ToDo: implement once get_state function is available, until then always reconfigure
        # if self.dformat() == 1:
        #     file_extension = '_0.txt'
        # elif self.dformat() == 2:
        #     file_extension = '_0.bin'

        # filepath = self.FILE_LOCATION_FROM_CONTROL + filename + file_extension
            
        # with open(filepath, 'r') as f:
        #     num_elements = int(next(f).strip('\n'))
        #     total_num_samples = int(next(f).strip('\n'))

        # return [num_elements, total_num_samples]  # (number of elements, total number of samples per channel)
        return None

    def load_waveform_from_file(self, new_waveform_file=None):

        is_new_shape = self._is_new_waveform_shape(new_waveform_file)

        if new_waveform_file is not None:
            # update file and waveform shape if using new file
            self._set_file_mask(new_waveform_file)
            self._shape = self.get_waveform_shape(new_waveform_file)

        if self._state < 2 or is_new_shape:
            self._configure_hardware()
        elif self._state == 4:
            self._disable_output()

        # ToDo: It also needs to reconfigure in order to switch between text and binary formats

        self._load_waveform_to_fpga()

        self._state = 3

    def run(self, file=None):
        # ToDo: with get_state function, should only load waveform if it hasn't already been loaded
        self.load_waveform_from_file(file)
        self._enable_output()

    def stop(self):
        self._disable_output()

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

        """
        Assumptions:
            1. All elements have the same channels outputting in them, so the channels in seq[0] are the same as in
               every other element.
            2. The length (in samples) of the longest channel output array for element 0 is also the length of the 
               longest output array for the entire sequence.
            3. The total number of samples (i.e. block or element size) is the same for all channels

        """

        used_channels = [ch for ch in seq[0]['data'].keys()]
        for ch in used_channels:
            if ch not in range(1, self.NUM_CHANNELS+1):
                warnings.warn(f"Unknown channel specified: {ch}. AWG has channels 1-{self.NUM_CHANNELS}. "
                              f"Data for {ch} will not be uploaded.")

        # get element size (size of longest channel output array)
        block_size = max([len(a) for a in seq[0]['data'].values()])  # Assumption 2

        # create output dictionary containing list of output data for all channels, including padding on each element
        output_dict = {ch: [] for ch in range(1, self.NUM_CHANNELS + 1)}
        for element in seq:
            for ch in output_dict:
                for rep in range(element['sequencing']['nrep']):
                    if ch in element['data']:
                        # ToDo: convert output dict values into twos complement data here
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

        data = self._make_file_data(n_blocks, padded_block_size, output_dict)

        self._create_files(header, data, filename)

    def _make_file_header(self, n_blocks, total_num_samples, channels_per_file=8):
        """args: number of elements, total number of samples
        returns: IO for file header, as either a StringIO or BytesIO, depending on current data_format
        """

        header = self._data_object()

        # if binary, header data is 4 byte unsigned integers, instead of default 2 byte signed used for remaining data
        self.write_sample(header, n_blocks, self.data_format(), b=4, signed=False)
        for i in range(channels_per_file):
            self.write_sample(header, total_num_samples, self.data_format(), b=4, signed=False)

        return header

    def _make_file_data(self, n_blocks, padded_block_size, output_dict):

        # ToDo: it feels like there must be a better way to organize the output_dict so that this is better

        data = {}

        for file_i, channel_list in self.FILE_CHANNEL_MAPPING.items():
            file_output_array = [output_dict[ch] for ch in channel_list]

            output = self._data_object()

            for i_block in range(n_blocks):
                for i_sample in range(padded_block_size):  # Assumption 3
                    for i_channel in range(len(channel_list)):
                        a = file_output_array[i_channel][i_block]
                        try:
                            current_sample = int(a[i_sample])
                        except IndexError:
                            current_sample = int(0)
                        self.write_sample(output, current_sample, self.data_format())

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
                shutil.copyfileobj(contents, fd)

    @staticmethod
    def write_sample(stream, sample, dformat, b=2, signed=True):
        if dformat == 'TXT':
            print('{}'.format(sample), file=stream)
        elif dformat == 'BIN':
            stream.write(sample.to_bytes(b, byteorder=sys.byteorder, signed=signed))

    @classmethod
    def forged_seq_array_to_16b2c(cls, array):
        """Takes an array with values between -1 and 1, where 1 specifies max voltage and -1 min voltage.
        Converts the array into twos-complement data, as required by the AWG."""
        return (array * cls.max_16b2c).astype(int)

    @classmethod
    def _voltage_to_int(cls, v):
        return int(round(v / cls.V_PP_DC * 2 ** (cls.DAC_RESOLUTION_BITS - 1)))
