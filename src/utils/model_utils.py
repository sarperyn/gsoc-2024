import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

def save_model(model, filepath):
    torch.save(model, filepath)

def load_model(filepath):
    return torch.load(filepath)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

def set_model(model_class, model_dir, device):
    model = model_class()
    model = torch.load(model_dir).to(device)
    return model

def sample_from_vae(model, num_samples, device, latent_dim=20):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_samples = model.decode(z)
        
        return generated_samples

def sample_from_diff(model, sample_shape, num_samples, device):
    model.eval()  
    with torch.no_grad(): 
        xt = torch.randn((num_samples, *sample_shape)).to(device)
        for t in reversed(range(model.num_timesteps)):
            t_tensor = torch.tensor([t] * num_samples, device=device).long()
            xt, _ = model.reverse_diffusion(xt, t_tensor)
        return xt

