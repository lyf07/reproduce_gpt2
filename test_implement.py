import model
import torch
from transformers import GPT2LMHeadModel
import tiktoken

def copymodel(model):
    new_model = GPT2LMHeadModel.from_pretrained("gpt2")
    new_l = [k for k in new_model.state_dict().keys()]
    l = [k for k in model.state_dict().keys() if not(k.endswith('mask'))]
    linear_layers = ['c_attn.weight', 'c_proj.weight', 'c_fc.weight', 'c_proj.weight']
    # print(len(l))
    # for k, v in new_model.state_dict().items():
    #     print("new: ", v)
        # break
    for i in range(len(new_l)):
        assert(new_model.state_dict()[new_l[i]].shape == model.state_dict()[l[i]].shape or new_model.state_dict()[new_l[i]].shape == model.state_dict()[l[i]].t().shape)
        if any(new_l[i].endswith(layer) for layer in linear_layers):
            # print(f"new_l: {new_l[i]}, l: {l[i]}")
            with torch.no_grad():
                # model.state_dict()[l[i]] = new_model.state_dict()[new_l[i]].t()
                model.state_dict()[l[i]].copy_(new_model.state_dict()[new_l[i]].t())
        else:
            with torch.no_grad():
                model.state_dict()[l[i]].copy_(new_model.state_dict()[new_l[i]])
                
def test_load_model():
    config = model.GPTConfig()
    gpt = model.GPT(config)
    copymodel(gpt)
    text = "Hello, I'm a language model,"
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(5, 1)
    # print(tokens.shape)
    response = gpt.generate(tokens, max_length=30)
    
    for i in range(5):
        print(encoder.decode(response[i].tolist()))
    
    
def test_loss():
    config = model.GPTConfig()
    gpt = model.GPT(config)
    data = [1]
    target = [6]
    # data = torch.tensor(data, dtype=).unsqueeze(0)
    # train = create_dataset()
    data = torch.tensor(data).unsqueeze(0)
    print(data.shape)
    target = torch.tensor(target).unsqueeze(0)
    response, loss = gpt.forward(data, target)
    # should be around 10 here due to cross entropy loss
    print(loss.detach())
    
if __name__ == '__main__':
    # test_load_model()
    # test_loss()
    config = model.GPTConfig()
    gpt = model.GPT(config)
    for k, v in gpt.state_dict().items():
        print(k, v.shape)