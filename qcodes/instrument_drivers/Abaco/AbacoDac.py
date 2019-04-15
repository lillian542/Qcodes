from contextlib import contextmanager
import numpy as np
import io
from typing import List
from math import ceil
import shutil
import time
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
    FILENAME = "test{}.{}"
    FILE_LOCATION = "//DESKTOP-LUEGMM9/Abaco_4DSP_waveforms/"

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

    def __init__(self, name, address, port, *args, **kwargs) -> None:
        # address is TCPIP0::hostname::port::SOCKET
        # self._visa_address = "TCPIP0::{:s}::{:d}::SOCKET".format(address, port)
        # super().__init__(name, self._visa_address, terminator='', **kwargs)
        super().__init__(name, address, port, *args, **kwargs, persistent=False, terminator='')
        # with self.temporary_timeout(11):
        #     print("asked returned {}".format(self.ask("init_state\n")))
        #     print("asked returned {}".format(self.ask("init_state\n")))
        # # cls.ask("init_state")
        # # time.sleep(1)
        # # cls.ask("config_state")
        # # glWaveFileMask=test_
        # pass

        self._state = 2
        self._shape = {}

        #self._initialize()
        # ToDo: it would be better if it only initialized if it weren't already initialized - its time consuming and restarting the kernel doesn't require reinitializing the instrument
        #self._specify_file(file='initial_file')

        print("Abaco connected")

        # ToDo: check that the file for the waveforms can be located, useful error if not

    #     # ToDo: is this ridiculous and what does it mean anyway? And it should start as whatever the default will be...
    #     self.add_parameter('voltage_coupling_mode',
    #                        set_cmd=self.set_V_pp,
    #                        get_cmd=self.get_V_pp,
    #                        vals=vals.Enum('DC', 'AC'))
    #     print(self.voltage_coupling_mode)
    #     self.voltage_coupling_mode.set('DC')
    #     print(self.voltage_coupling_mode)
    #     # ToDo: this, like voltage coupling mode, needs to start out with some default
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

    def _initialize(self):
        self.ask('init_state')
        time.sleep(80)  # ToDo: do this with temporary timeout instead? How do timeouts work for instruments?
        self._state = 1

    def _configure_hardware(self):
        self.ask('config_state')
        time.sleep(60)
        self._state = 2

    def _specify_file(self, file):
        # ToDo: talk to Ruben about how to specify which file to upload without accessing the GUI
        pass

    def _load_waveform_to_fpga(self):
        self.ask('load_waveform_state')
        self._state = 3
        self._shape = {}  # ToDo: add shape!

    def load_waveform(self, new_waveform=None):
        if new_waveform is not None:
            self._specify_file(new_waveform)

        if self._state < 2 or self._is_new_waveform_shape(new_waveform):
            self._configure_hardware()
        elif self._state == 4:
            self._disable_output()

        self._load_waveform_to_fpga()

        self._state = 3
        self._shape = {}  # ToDo: add shape!

    def _enable_output(self):
        if self._state != 3:
            raise RuntimeError('Waveform not uploaded, cannot enable output') # ToDo: change this to reupload current file?

        self.ask('enable_offload_state')

        self._state = 4

    def _disable_output(self):
        if self._state != 4:
            raise RuntimeError("Waveform output not enabled, cannot disable output")

        self.ask('disable_offload_state')

        self._state = 5


    @contextmanager
    def temporary_timeout(self, timeout):
        old_timeout = self._timeout
        self.set_timeout(timeout)
        yield
        self.set_timeout(old_timeout)

    def _is_new_waveform_shape(self, new_waveform):

        if new_waveform is None:
            return False
        
        current_shape = self._shape

        self.get_waveform_shape(new_waveform)

        # ToDo: compare current and new waveforms, return False if they have the same shape, else True

        return True

    def get_waveform_shape(filename):
        # ToDo: create this function (what format does the new waveform have? is it a forged sequence? a summary?)
        return {}

    def upload_to_fpga(self, file=None):
        # ToDo: select file to upload
        # reuploads last used waveform if no file is specified
        start = time.clock()
        new_shape = self._is_new_waveform_shape(file)

        # output must be disabled before any other change of intrument state can occur
        self._disable_output()

        self._specify_file_for_upload(file)

        # must reinitialize and reconfigure hardware if the new waveform does not have the same basic shape
        # shape: (number of samples, number of blocks)
        if new_shape:
            self._initialize()
            time.sleep(80)
            self._hardware_configure()
            

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
        # ToDo: should this disable output or just stop the triggers?
        # If it disables output, you need to go back to upload before you can run again.

    ######################
    # AWG file functions #
    ######################

    @classmethod
    def _create_files(cls, header, data, dformat, filename=None):
        if filename is None:
            filepath = cls.FILE_LOCATION + cls.FILENAME
        else:
            filepath = cls.FILE_LOCATION + filename + '{}.{}'

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
            contents = content_type()
            contents.write(header.getvalue())
            contents.write(data[i].getvalue())

            with open((filepath).format(i, file_type), file_access) as fd:
                contents.seek(0)
                shutil.copyfileobj(contents, fd)

    def make_and_save_awg_file_locally(self, seq: List[np.ndarray], dformat: int, filename=None):
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

        # ToDo: currently the forged sequences are output with both channels and markers.
        #       The marker names in the forged sequence will need to be converted to channel numbers to upload here

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

    @staticmethod
    def write_sample(stream, sample, dformat, bytes=2, signed=True):
        if dformat == 1:
            print('{}'.format(sample), file=stream)
        elif dformat == 2:
            stream.write(sample.to_bytes(bytes, byteorder=sys.byteorder, signed=signed))
