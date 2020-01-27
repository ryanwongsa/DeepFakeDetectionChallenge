class MetricLogger(object):
    def __init__(self):
        self.dict_metrics = {}
        self.dict_metric_steps = {}
    
    def update_metric(self, value, name):
        self.dict_metrics[name] = value
    
    def increment_metric(self, value, name):
        m_v = self.dict_metrics[name]
        m_v += (value - m_v)/(self.dict_metric_steps[name]+1)
        self.dict_metrics[name] = m_v
        self.dict_metric_steps[name] += 1
    
    def reset_metrics(self, names):
        for name in names:
            self.dict_metrics[name] = 0
            self.dict_metric_steps[name] = 0
    
    def get(self, name):
        return self.dict_metrics[name]