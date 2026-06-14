from Experiment import SNNExperiment
from Encoding import *
import time

"""
Archivo para analizar los spikes generados por cada codificador y
ver la media de spikes generados por imagen y por batch
"""

class EncoderAnalysis:
    def __init__(self, config):
        self.config = config

    def run_experiment(self):
        #print("Sacar un batch del dataset para analizar los spikes generados por cada codificador")

        train_loader, test_loader = SNNExperiment.init_dataset(self).get_loaders()
        
        steps = [5, 10, 25, 50, 100, 200]
        
        spikes_totales = 0
        imagenes_totales = 0
        
        for encoder_name in ['rate', 'ttfs', 'Deterministic', 'delta', 'MW', 'SF', 'direct']:
            
            print(f"\n\n\nAnalizando el codificador :")
            print("\nAnalizar spikes generados para todo el dataset:")
            
            #tiempo total que tarda en codificar todo cada codificador
            
            time_start = time.time()
            
            for step in steps:
                
                self.config['num_steps'] = step
                self.config['encoder'] = encoder_name
                encoder = SNNExperiment.init_encoder(self)
                                
                for data, targets in test_loader:
                    
                    dataEncoded = encoder.encode(data)
                    spikes_totales += dataEncoded.sum()
                    imagenes_totales += targets.size(0)
                
                print(f"\n{encoder_name.upper()} con {step} pasos temporales: {spikes_totales} spikes")
                print(f"Media de spikes por imagen: {spikes_totales / imagenes_totales}")
                print(f"Media de spikes por batch: {spikes_totales / graphics_config['batch_size']}")
            
            time_end = time.time()
            print(f"\nTiempo total para codificar todo el dataset con {encoder_name.upper()}: {time_end - time_start:.2f} segundos")
        
 

# Base configuration for the experiment
graphics_config = {
    'dataset': 'MNIST',
    'data_path': './data/mnist',
    'batch_size': 128,
    'num_steps': 25,
    }

if __name__ == "__main__":
    analysis = EncoderAnalysis(graphics_config)
    analysis.run_experiment()
    
