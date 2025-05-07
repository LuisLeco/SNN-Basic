import time
from Experiment import SNNExperiment
import csv


num_steps = [5,10,30,50,100,200]

config = {
    'dataset': 'MNIST',
    'encoder': 'ttfs_time', # posibles: poisson, rate, ttfs, direct, ttfs_time, delta, MW, SF
    'decoder': 'rate', # posibles: rate, latency, first_spike, population,rank_order
    'architecture': 'TwoLayerSNN',
    'data_path': './data/mnist',
    'batch_size': 128,
    'num_inputs': 784,
    'num_hidden': 1000,
    'num_outputs': 10,
    'num_steps': 25,
    'beta': 0.95,
    'lr': 5e-4,
    'betas': (0.9, 0.999),
    'num_epochs': 1, #Tiene que ser 32
    'eval_freq': 1,
    'decoder_params': {
        'rate': {'scale': 1.0},
        'latency': {'target_time': 0.5, 'sensitivity': 1.0},
        'first_spike': {'threshold': 0.1},
        'population': {'num_classes': 10,'num_neurons_per_class': 1},
        'rank_order': {'num_classes': 10}
    }}

if __name__ == "__main__":
    start_time = time.time()
    with open(f'results {config['encoder']}.csv', 'w', newline='') as csvfile:
        fieldnames = ['encoder', 'decoder', 'num_steps', 'accuracy', 'numero de prueba']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Bucle de num_steps
        for i in num_steps:
            config['num_steps'] = i
            
            #Ejecutar 5 veces cada experimento
            for trial in range(1,6):
                experiment = SNNExperiment(config)
                final_accuracy = experiment.run()
                writer.writerow({
                    'encoder': config['encoder'],
                    'decoder': config['decoder'],
                    'num_steps': config['num_steps'],
                    'accuracy': f"{final_accuracy:.2f}",
                    'numero de prueba': trial
                })
                print(f"num_steps={i}, trial={trial} → Final accuracy: {final_accuracy:.2f}%")                
                
        end_time = time.time()
        total_time = end_time - start_time

        # Aquí añadimos la línea de comentario al final del CSV
        csvfile.write(f"# Tiempo total de ejecucion: {total_time:.4f} segundos\n") # revisar el \n del final y ver si funciona para ponerlo tambien al principio
    
    print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")