# Importando as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Imprimindo a versão do TensorFlow para referência
print(f"TensorFlow version: {tf.__version__}")

def create_model():
    """
    Cria e retorna um modelo de rede neural convolucional (CNN) para classificação binária de imagens.
    
    Arquitetura do modelo:
    - Três camadas convolucionais com MaxPooling
    - Uma camada Flatten
    - Duas camadas Dense (totalmente conectadas)
    
    Returns:
        Um modelo Keras compilado
    """
    model = models.Sequential([
        # Camada de entrada: 32 filtros, kernel 3x3, ativação ReLU, para imagens 224x224 com 3 canais
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Terceira camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Flatten para transformar o volume 3D em um vetor 1D
        layers.Flatten(),
        # Camada densa com 64 neurônios
        layers.Dense(64, activation='relu'),
        # Camada de saída para classificação binária
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilando o modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def infinite_generator(generator):
    """
    Cria um gerador infinito a partir de um gerador existente.
    
    Args:
        generator: Um gerador de dados Keras
    
    Yields:
        Lotes de dados indefinidamente
    """
    while True:
        for batch in generator:
            yield batch

def train_model(model, train_dir, val_dir, batch_size=32, epochs=20):
    """
    Treina o modelo usando os dados nos diretórios especificados.
    
    Args:
        model: O modelo Keras a ser treinado
        train_dir: Caminho para o diretório de dados de treinamento
        val_dir: Caminho para o diretório de dados de validação
        batch_size: Tamanho do lote para treinamento
        epochs: Número de épocas para treinar
    
    Returns:
        O histórico de treinamento
    """
    # Configurando os geradores de dados com aumento de dados
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Criando geradores de fluxo de diretório
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    # Calculando o número de passos por época
    train_steps_per_epoch = train_generator.samples // batch_size
    val_steps_per_epoch = val_generator.samples // batch_size

    # Usando geradores infinitos para garantir que não haja interrupções durante o treinamento
    train_generator = infinite_generator(train_generator)
    val_generator = infinite_generator(val_generator)

    # Treinando o modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch)

    return history

if __name__ == "__main__":
    # Configurando os caminhos dos diretórios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    train_dir = os.path.join(project_dir, 'data', 'train')
    val_dir = os.path.join(project_dir, 'data', 'val')
    models_dir = os.path.join(project_dir, 'models')
    
    # Caminhos alternativos (comentados)
    # train_dir = 'caminho/para/seu/diretorio/de/treinamento'
    # val_dir = 'caminho/para/seu/diretorio/de/validacao'
    
    # Criando e treinando o modelo
    model = create_model()
    history = train_model(model, train_dir, val_dir, batch_size=32, epochs=20)

    # Salvando o modelo final
    model.save('modelo_final.h5')

    print("Treinamento concluído. O modelo final foi salvo como 'modelo_final.h5'.")