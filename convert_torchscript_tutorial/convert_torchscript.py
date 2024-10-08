import torch
import torchvision


def convert_to_torchscript_via_tracing():
    # An instance of your model.
    model = torchvision.models.resnet18()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example) # ScriptModule

    return traced_script_module




class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

def convert_to_torchscript_via_annotation():
    my_module = MyModule(10,20)
    sm = torch.jit.script(my_module) # ScriptModule
    return sm




def serialising(script_module, filename):
   script_module.save(filename)


if __name__ == "__main__":

    traced_script_module = convert_to_torchscript_via_tracing()
    serialising(traced_script_module, 'converted_resnet_model.pt')

    # scripted_module = convert_to_torchscript_via_annotation()
    # serialising(scripted_module, "converted_module.pt")