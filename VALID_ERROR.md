REMOVE SHUTIL

Best validation loss: 0.018000158248111034
Saving best model for epoch 25

Learning rate adjusted to: 0.00027738957312183364
Model saved to output/2024_11_13_22_18_32/model.pth
Training failed with error: unhashable type: 'dict'
Traceback (most recent call last):
  File "train_deeplabv3.py", line 165, in <module>
    raise e
  File "train_deeplabv3.py", line 161, in <module>
    train(config)
  File "train_deeplabv3.py", line 126, in train
    metric_tracker.save_plots()
  File "/home/minhqvu/segmentation-module/deeplabv3_apples/validate.py", line 100, in save_plots
    
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/pyplot.py", line 2769, in plot
    **({"data": data} if data is not None else {}), **kwargs)
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/axes/_axes.py", line 1635, in plot
    lines = [*self._get_lines(*args, data=data, **kwargs)]
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/axes/_base.py", line 312, in __call__
    yield from self._plot_args(this, kwargs)
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/axes/_base.py", line 495, in _plot_args
    self.axes.yaxis.update_units(y)
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/axis.py", line 1449, in update_units
    default = self.converter.default_units(data, self)
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/category.py", line 116, in default_units
    axis.set_units(UnitData(data))
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/category.py", line 192, in __init__
    self.update(data)
  File "/opt/conda/lib/python3.7/site-packages/matplotlib/category.py", line 225, in update
    for val in OrderedDict.fromkeys(data):
TypeError: unhashable type: 'dict'
