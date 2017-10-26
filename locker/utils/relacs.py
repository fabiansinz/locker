import re
import yaml
import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()

def scan_info(cell_id, basedir):
    """
    Scans the info.dat for meta information about the recording.

    Args:
        cell_id: id of the cell
        basedir: base directory where the data lives


    """
    info = open(basedir + cell_id + '/info.dat').readlines()
    info = [re.sub(r'[^\x00-\x7F]+', ' ', e[1:]) for e in info]
    meta = yaml.load(''.join(info))
    return meta


def get_number_and_unit(value_string):
    """
    Get the number and the unit from a string.

    :param value_string: string with number and unit
    :return: value, unit
    """
    if value_string.endswith('%'):
        return (float(value_string.strip()[:-1]), '%')
    try:
        a = ureg.parse_expression(value_string)
    except:
        return (value_string, None)

    if type(a) == list:
        return (value_string, None)

    if isinstance(a, (int, float)):
        return (a, None)
    else:
        # a.ito_base_units()
        value = a.magnitude
        unit = "{:~}".format(a)
        unit = unit[unit.index(" "):].replace(" ", "")

        if unit == 'min':
            unit = 's'
            value *= 60
        return (value, unit)


def load_traces(relacsdir, stimuli):
    """
    Loads trace files from relacs data directories,

    :param relacsdir: directory where the traces are stored
    :param stimuli: stimuli file object from pyrelacs
    :return: dictionary with loaded traces
    """
    meta, key, data = stimuli.selectall()

    ret = []
    for _, name, _, data_type, index in [k for k in key[0] if 'traces' in k]:
        tmp = {}
        sample_interval, time_unit = get_number_and_unit(meta[0]['analog input traces']['sample interval%i' % (index,)])
        sample_unit = meta[0]['analog input traces']['unit%i' % (index,)]
        x = np.fromfile('%s/trace-%i.raw' % (relacsdir, index), np.float32)

        tmp['unit'] = sample_unit
        tmp['trace_data'] = name
        tmp['data'] = x
        tmp['sample_interval'] = sample_interval
        tmp['sample_unit'] = sample_unit
        ret.append(tmp)
    return {e['trace_data']: e for e in ret}