import numpy as np
from lomentum import SegmentGroup, Element
from lomentum.atoms import zero, marker_on, marker_off, marker_pulse
from qdev_wrappers.pulse_building.atoms_ext import sine_multi

def create_readout_template_element():
    seg1 = zero(duration='drive_stage_duration')
    seg2 = sine_multi(frequencies='readout_sidebands', duration='readout_duration',
                      amplitudes='readout_amplitude', phases=np.pi/2)
    seg3 = zero(duration='after_readout_duration')

    readout_pulse_I = SegmentGroup(seg1, seg2, seg3,
                                   duration='total_duration')

    seg1 = zero(duration='drive_stage_duration')
    seg2 = sine_multi(frequencies='readout_sidebands', duration='readout_duration',
                      amplitudes='readout_amplitude')
    seg3 = zero(duration='after_readout_duration')

    readout_pulse_Q = SegmentGroup(seg1, seg2, seg3,
                                   duration='total_duration')

    m1 = marker_off(duration='pre_marker_duration')
    m2 = marker_on(duration='marker_duration')
    m3 = marker_off(duration='post_marker_duration')

    markers = SegmentGroup(m1, m2, m3,
                           duration='total_duration')

    def mytransformation(context):
        context['after_readout_duration'] = context['total_duration'] - \
            context['drive_stage_duration'] - context['readout_duration']
        context['pre_marker_duration'] = context['drive_stage_duration'] - \
            context['marker_readout_delay']
        context['post_marker_duration'] = context['total_duration'] - \
            context['marker_duration'] - context['pre_marker_duration']
        context['readout_sidebands'] = [v for k, v in context.items() if k.startswith('readout_sideband')]

    template_element = Element(segments={3: readout_pulse_I,
                                         4: readout_pulse_Q,
                                         '4M1': markers,
                                         '4M2': zero(duration='total_duration')},
                               sequencing={'nrep': 1},
                               transformation=mytransformation)
    return template_element
