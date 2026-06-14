import torch
import numpy as np
import matplotlib.pyplot as plt
from Experiment import SNNExperiment


"""
Clase para generar los gráficos de los spikes generados por cada
codificador y analizar el impacto de los parametros de la red.
"""

class GraphicsPlot:
    def __init__(self, tensor_original, tensor_spikes, targets, nombre_codificador, batch_idx=0, channel_idx=0):
        self.img_analogica = tensor_original[batch_idx, channel_idx].detach().cpu().numpy()
        self.spikes_img = tensor_spikes[:, batch_idx, channel_idx, :, :].detach().cpu().numpy()
        self.num_steps = self.spikes_img.shape[0]
        
        self.etiqueta = targets[batch_idx].item()
        # 2. Guardamos el nombre
        self.nombre_codificador = nombre_codificador
        
        self.coordenadas =[(14, 14), (10, 14), (7, 14), (2, 2)]
        self.colores = plt.cm.tab10.colors
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1, 3, figsize=(16, 5), gridspec_kw={'width_ratios':[1, 1, 1.5]}
        )
        
        # 3. Ponemos el nombre del codificador y el target en el súper título superior
        self.fig.suptitle(
            f"Análisis de Píxeles | Codificador: {self.nombre_codificador} | Etiqueta Real: {self.etiqueta} | Num Steps: {self.num_steps}", 
            fontsize=16, 
            fontweight='bold'
        )
        
        self.fig.canvas.mpl_connect('button_press_event', self.al_hacer_clic)
        
        self.actualizar_graficas()
        plt.tight_layout()
        plt.show()

    def al_hacer_clic(self, event):
        # (El código de al_hacer_clic se queda exactamente igual)
        if event.inaxes == self.ax1:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            alto, ancho = self.img_analogica.shape
            if 0 <= x < ancho and 0 <= y < alto:
                self.coordenadas.pop(0)
                self.coordenadas.append((y, x))
                self.actualizar_graficas()
                print(f"Coordenadas seleccionadas: (y={y}, x={x}) → Valor analógico: {self.img_analogica[y, x]}")

    def actualizar_graficas(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        num_pixeles = len(self.coordenadas)
        valores_analogicos = []
        tiempos_spikes =[]
        
        for y, x in self.coordenadas:
            valores_analogicos.append(self.img_analogica[y, x])
            historia_pixel = self.spikes_img[:, y, x]
            tiempos_spikes.append(np.where(historia_pixel > 0)[0])

        # --- PANEL 1: Imagen ---
        self.ax1.imshow(self.img_analogica, cmap='gray')
        for i, (y, x) in enumerate(self.coordenadas):
            self.ax1.plot(x, y, marker='o', color=self.colores[i], markersize=12, markeredgecolor='white', markeredgewidth=2)
            
        # 3. También lo puedes poner en el título de la imagen si lo prefieres
        self.ax1.set_title(f"Imagen Original (Target: {self.etiqueta})")
        self.ax1.axis('off')

        # --- PANEL 2 y 3 se quedan IGUAL ---
        posiciones_y = np.arange(num_pixeles)
        barras = self.ax2.barh(posiciones_y, valores_analogicos, color=self.colores[:num_pixeles], height=0.4)
        self.ax2.set_xlim(0, 1.1)
        self.ax2.invert_yaxis()
        self.ax2.set_yticks(posiciones_y)
        self.ax2.set_yticklabels([f"Píxel {i}\n(y:{y}, x:{x})" for i, (y, x) in enumerate(self.coordenadas)])
        self.ax2.set_title("2. Valor Input (P)")
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        for barra in barras:
            self.ax2.text(barra.get_width() + 0.05, barra.get_y() + barra.get_height()/2, 
                     f'{barra.get_width():.2f}', va='center', fontweight='bold')

        self.ax3.eventplot(tiempos_spikes, lineoffsets=posiciones_y, linelengths=0.6, colors=self.colores[:num_pixeles], linewidths=3.0)
        self.ax3.set_xlim(-0.5, self.num_steps - 0.5)
        self.ax3.invert_yaxis()
        self.ax3.set_title("3. Spikes generados en el tiempo")
        self.ax3.set_xlabel("Pasos temporales (Time steps)")
        self.ax3.set_yticks([])
        self.ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for y in posiciones_y:
            self.ax3.axhline(y, color='gray', linestyle=':', alpha=0.5, zorder=0)

        self.fig.canvas.draw_idle()

class GraficosExperiment:
    def __init__(self, config):
       self.config = config
       self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
       self.dataset = SNNExperiment.init_dataset(self)
       self.encoder = SNNExperiment.init_encoder(self)
    
    def run(self):
       graphic_loader = self.dataset.get_loaders()
       train_loader = graphic_loader[0]
       
       data, targets = next(iter(train_loader))
       
       dataCodificada = self.encoder.encode(data).to(self.device)
       targets = targets.to(self.device)
       
       # 1. Extraemos el nombre del codificador desde tu configuración (ej: 'ttfs')
       # (Usamos .upper() para que quede en mayúsculas y más estético, ej: 'TTFS')
       nombre_codificador = str(self.config['encoder']).upper()
       
       # 2. Pasamos 'nombre_codificador' como cuarto argumento
       grafica = GraphicsPlot(data, dataCodificada, targets, nombre_codificador)


# Base configuration for the experiment
graphics_config = {
    'dataset': 'MNIST',
    'encoder': 'rate', # possible: poisson, rate, ttfs, direct, delta, MW, SF, Deterministic
    'data_path': './data/mnist',
    'batch_size': 1,
    'num_steps': 25,
    }

if __name__ == "__main__":
    print(f"Graficos para el codificador: {graphics_config['encoder']}")
    experiment = GraficosExperiment(graphics_config).run()