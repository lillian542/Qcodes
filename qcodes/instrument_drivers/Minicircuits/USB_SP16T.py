import os
from typing import Optional

# QCoDeS imports
from qcodes.instrument_drivers.Minicircuits.Base_SPDT import (
    SPDT_Base, SwitchChannelBase)

try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install pythonnet'
                         in a qcodes environment terminal""")


class SwitchChannelSP16T(SwitchChannelBase):
    def _set_switch(self, channel_num):
        command = ":SP16T:STATE:{}".format(channel_num)
        self._parent.switch.Send_SCPI(command, '')

    def _get_switch(self):
        return self._parent.switch.Send_SCPI(":SP16T:STATE?", '')[1]


class USB_SP16T(SPDT_Base):
    """
    Mini-Circuits SP16T RF switch

    Args:
            name: the name of the instrument
            driver_path: path to the dll
            serial_number: the serial number of the device
               (printed on the sticker on the back side, without s/n)
            kwargs: kwargs to be passed to Instrument class.
    """

    CHANNEL_CLASS = SwitchChannelSP16T
    PATH_TO_DRIVER = r'mcl_SolidStateSwitch64'

    def __init__(self, name: str, driver_path: Optional[str]=None, 
        serial_number: Optional[str]=None, **kwargs):
        # we are eventually overwriting this but since it's called
        # in __getattr__ of `SPDT_Base` it's important that it's
        # always set to something to avoid infinite recursion
        self._deprecated_attributes = None
        # import .net exception so we can catch it below
        # we keep this import local so that the module can be imported
        # without a working .net install
        clr.AddReference('System.IO')
        from System.IO import FileNotFoundException
        super().__init__(name, **kwargs)
        if os.name != 'nt':
            raise ImportError("""This driver only works in Windows.""")
        try:
            if driver_path is None:
                clr.AddReference(self.PATH_TO_DRIVER)
            else:
                clr.AddReference(driver_path)

        except (ImportError, FileNotFoundException):
            raise ImportError(
                """Load of mcl_SolidStateSwitch64.dll not possible. Make sure 
                the dll file is not blocked by Windows. To unblock right-click
                the dll to open properties and check the 'unblock' checkmark
                in the bottom. Check that your python installation is 64bit."""
            )
        import mcl_SolidStateSwitch64
        self.switch = mcl_SolidStateSwitch64.USB_Digital_Switch()                

        if not self.switch.Connect(serial_number)[0]:
            raise RuntimeError('Could not connect to device')
        self.address = self.switch.Get_Address()
        self.serial_number = self.switch.Read_SN('')[1]
        self.connect_message()
        self.add_channels(num_options=16)

    def get_idn(self):       
        fw = self.switch.Send_SCPI(':FIRMWARE?', '')[1]   
        MN = self.switch.Read_ModelName('')[1]   
        SN = self.switch.Read_SN('')[1]

        id_dict = {
            'firmware': fw,
            'model': MN,
            'serial': SN,
            'vendor': 'Mini-Circuits'
        }
        return id_dict

