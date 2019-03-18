'''
Tensorboard logging
'''
import tensorflow as tf
#from StringIO import StringIO
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os 
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, friendly_name):
        """Creates a summary writer logging to log_dir."""
        now = datetime.datetime.now()

        event_folder = os.path.join(log_dir, friendly_name)
        event_folder = os.path.join(event_folder, now.strftime("%Y-%m-%d %H:%M"))
        self.writer = tf.summary.FileWriter(event_folder)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        #self.writer.flush()