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
    FILENAME = "test"
    FILE_LOCATION_FROM_CONTROL = "//DESKTOP-LUEGMM9/Abaco_4DSP_waveforms/"
    FILE_LOCATION_FROM_AWG = "C:\\Abaco_4DSP_waveforms\\"

    FILE_CHANNEL_POSITION = {
        '_0': [2, 6, 10, 14, 1, 5, 9, 13],
        '_1': [4, 8, 12, 16, 3, 7, 11, 15]
    }

    NUM_CHANNELS = 8 * len(FILE_CHANNEL_POSITION)

    STATES = {0: 'not_initialized',
              1: 'initialized',
              2: 'hardware_configured',
              3: 'wavefrom_uploaded',
              4: 'output_enabled',
              5: 'output_disabled'}  # This seems not useful for anything except reading the code about the order of the functions more easily

    max_16b2c = 32767

    def __init__(self, name, address, port, initial_file='initial_file', dformat=1, new_initialization=True, **kwargs) -> None:
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

        if not os.path.exists(self.FILE_LOCATION_FROM_CONTROL):
            raise RuntimeError(f"The specified waveform file location, {self.FILE_LOCATION_FROM_CONTROL}, does not exist.")

        # ToDo: decide on shape for initial file
        self.dformat=dformat
        self.file_extensions = {1: '.txt', 2: '.bin'}
        self.ask(f'glWaveFileExtension={self.file_extensions[dformat]}')
        self.ask(f'glWaveFileFolder={self.FILE_LOCATION_FROM_AWG}')
        self.select_file(initial_file)

        if new_initialization:
            # ToDo: decide whether to initialize/configure based on get_state function
            self._initialize()
            self._configure_hardware()

        self._state = 2
        self.load_waveform_from_file()

        self._shape = self.get_waveform_shape(initial_file)

        self.add_parameter('max_trigger_freq', 
                           unit = 'Hz',
                           get_cmd=self._get_max_trigger_freq)

        print("Abaco connected")

    #     # ToDo: add option to set amplitude and offset by channel
    #     self.add_parameter('voltage_coupling_mode',
    #                        set_cmd=self.set_V_pp,
    #                        get_cmd=self.get_V_pp,
    #                        vals=vals.Enum('DC', 'AC'))
    #     print(self.voltage_coupling_mode)
    #     self.voltage_coupling_mode.set('DC')
    #     print(self.voltage_coupling_mode)
    #     self.add_parameter('V_pp',
    #                        set_cmd=self.set_V_pp,
    #                        get_cmd=self.get_V_pp,
    #                        label='Voltage peak-to-peak',
    #                        vals=vals.Numbers(0, self.MAX_V_PP[self.voltage_coupling_mode]))
    # def set_V_pp(self, val):
    #     self.V_pp = val
    # def get_V_pp(self):
    #     return self.V_pp
    # def set_voltage_coupling_mode(self, val):
    #     self.voltage_coupling_mode = val
    # def get_voltage_coupling_mode(self):
    #     return self.voltage_coupling_mode

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

    def _set_file_type(self, dformat):
        if dformat is None:
            dformat=self.dformat
        self.ask(f'glWaveFileExtension={self.file_extensions[dformat]}')
        # ToDo: I belive it needs to reconfigure if dformat is changed

    def _load_waveform_to_fpga(self):
        self.ask('load_waveform_state')
        self._state = 3
        self._shape = {}  # ToDo: add shape!

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
        max_data_rate_per_channel = (12.16/8) * 10e9  # 4DSP verification records pg 16

        return {'waveform_size_bytes': waveform_size_bytes, 'max_trigger_freq': int(max_data_rate_per_channel/waveform_size_bytes)}

    def _is_new_waveform_shape(self, new_waveform):

        if new_waveform is not None:
            new_shape = self.get_waveform_shape(new_waveform)

            if new_shape != self._shape:
                return True

        return False

    @classmethod
    def get_waveform_shape(cls, filename, dformat=1):
        # ToDo: rewrite this to get shape from get_state function
        filepath = cls.FILE_LOCATION_FROM_CONTROL + filename + '_0.txt'

        if dformat == 2:
            raise RuntimeError('Not sure how to extract shape from binary data file yet')
            
        with open(filepath, 'r') as f:
            num_elements = int(next(f).strip('\n'))
            total_num_samples = int(next(f).strip('\n'))

        return [num_elements, total_num_samples]  # (number of elements, total number of samples per channel)

    def load_waveform_from_file(self, new_waveform_file=None):
        if new_waveform_file is not None:
            self.select_file(new_waveform_file)

        if self._state < 2 or self._is_new_waveform_shape(new_waveform_file):
            self._configure_hardware()
        elif self._state == 4:
            self._disable_output()

        # ToDo: It also needs to reconfigure in order to switch between text and binary formats

        self._load_waveform_to_fpga()

        self._state = 3
        self._shape = {}  # ToDo: add shape!

    def select_file(self, filename, dformat=None):
        self._set_file_type(dformat)
        self._set_file_mask(filename)

    def run(self):
        self.load_waveform_from_file()
        self._enable_output()
        # ToDo: then start triggers? Or does run not make sense when the AWG only operates in external trigger mode?

    def stop(self):
        self._disable_output()
        # ToDo: should this disable output or just stop the triggers?

    ######################
    # AWG file functions #
    ######################

    def make_and_send_awg_file(self, seq: List[np.ndarray], dformat: int, filename=None):
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

        # ToDo: check dformat value is okay here, then pass it on to other things - or maybe pass on StringIO or ByteIO?

        header = self._make_file_header(n_blocks, total_num_samples, dformat)

        data = self._make_file_data(n_blocks, padded_block_size, output_dict, dformat)

        self._create_files(header, data, dformat, filename)

    @classmethod
    def _make_file_header(cls, n_blocks, total_num_samples, dformat: int, channels_per_file=8):
        """args: forged sequence
        returns: IO for file header, in either string (dformat=1) or binary (dformat=2) format.
        """
        if dformat == 1:
            header = io.StringIO()
        elif dformat == 2:
            header = io.BytesIO()
        else:
            raise RuntimeError(f"Variable dformat must be 1 (for txt file) or 2 (for bin file). Received {dformat}.")

        # if binary, header data is 4 byte unsigned integers, instead of default 2 byte signed used for remaining data
        cls.write_sample(header, n_blocks, dformat, bytes=4, signed=False)
        for i in range(channels_per_file):
            cls.write_sample(header, total_num_samples, dformat, bytes=4, signed=False)

        return header

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
            # ToDo: I think just check that dformat has a correct value once, instead of this over and over - but where?

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

    @classmethod
    def _create_files(cls, header, data, dformat, filename=None):
        if filename is None:
            filename = cls.FILENAME
        
        filepath = cls.FILE_LOCATION_FROM_CONTROL + filename + '{}.{}'

        if dformat == 1:
            file_access = 'w'
            file_type = 'txt'
            content_type = io.StringIO
        else:
            file_access = 'wb'
            file_type = 'bin'
            content_type = io.BytesIO

        # write files to disk
        for i in data:
            contents = content_type()
            contents.write(header.getvalue())
            contents.write(data[i].getvalue())

            with open(filepath.format(i, file_type), file_access) as fd:
                contents.seek(0)
                shutil.copyfileobj(contents, fd)

    @staticmethod
    def write_sample(stream, sample, dformat, bytes=2, signed=True):
        if dformat == 1:
            print('{}'.format(sample), file=stream)
        elif dformat == 2:
            stream.write(sample.to_bytes(bytes, byteorder=sys.byteorder, signed=signed))

    def forged_seq_array_to_16b2c(self, array):
        """Takes an array with values between -1 and 1, where 1 specifies max voltage and -1 min voltage.

        Returns an array of voltages (based on the current set peak-to-peak voltage) converted into twos-complement
        data, as required by the AWG."""
        # ToDo: fix this when adding setting peak-to-peak voltage
        # ToDo: make this channel specific, as peak to peak voltage could be set per channel
        try:
            amplitude_scaling = self.V_pp/self.MAX_V_PP[self.voltage_coupling_mode]
        except:
            amplitude_scaling = 1
        # ToDo: test this
        return (array * self.max_16b2c * amplitude_scaling).astype(int)

    @classmethod
    def _voltage_to_int(cls, v):
        return int(round(v / cls.V_PP_DC * 2 ** (cls.DAC_RESOLUTION_BITS - 1)))
