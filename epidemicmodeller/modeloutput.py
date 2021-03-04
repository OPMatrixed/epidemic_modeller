class ModelOutput(object):
    def __init__(self, _type, params, final_size, duration, data_log):
        self.type = _type
        self.params = params
        self.final_size = final_size
        self.duration = duration
        self.data_log = data_log
