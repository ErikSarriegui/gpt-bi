<p align="center" sty>
   <img src="images/gpt-bi-logo.jpg" width="450">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/license-MIT-green">
   
   <a href="https://huggingface.co/AuriLab/gpt-bi" target="_blank">
      <img src="https://img.shields.io/badge/HuggingFace-%F0%9F%A4%97-orange" />
   </a>
   
   <img src="https://img.shields.io/badge/Pretrained_Models-green">
   <!-- <img src="https://img.shields.io/badge/Blog%20Post-yellow"> SE PUEDE HACER UN POST EN MEDIUM -->
   <!-- <img src="https://img.shields.io/badge/Paper-blue"> Cuando haya un reporte técnico -->

   <a href="https://colab.research.google.com/github/ErikSarriegui/gpt-bi/blob/main/quickstart.ipynb" target="_blank">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
   </a>
</p>

# **GPT-Bi: Un modelo de lenguaje basado en Transformers para Euskera**

Introducimos GPT-Bi, un modelo de lenguaje abierto basado en la arquitectura de GPT-2 diseñado para experimentar con la generación de texto en euskera. Este proyecto tiene como objetivo principal avanzar en el campo del procesamiento del lenguaje natural (PLN) para lenguas minoritarias, como el euskera, que históricamente han tenido menos recursos y modelos de lenguaje disponibles en comparación con idiomas mayoritarios como el inglés o el español. El euskera es una lengua única, no indoeuropea, con una estructura gramatical y sintáctica compleja. A pesar de su riqueza cultural e histórica, los recursos digitales y herramientas de PLN para el euskera son limitados. Nuestra misión es reducir la brecha tecnológica y empoderar a individuos y organizaciones proporcionando un modelo de lenguaje ligero, robusto y accesible que pueda ser utilizado sin restricciones, animamos a la comunidad a contribuir con datos, entrenamiento y mejoras del modelo.

## **Quickstart**
Aquí te guiaremos para que puedas comenzar a utilizar este modelo diseñado específicamente para el euskera. Sigue estos pasos para explorar y experimentar con la generación de texto en euskera.

### **Probar la aplicación**
Puedes probar GPT-Bi directamente en tu navegador a través de nuestra demo en Hugging Face. Haz clic en el siguiente enlace para acceder a la interfaz y comenzar a generar texto en euskera:

👉 ![Demo de GPT-Bi](https://huggingface.co/spaces/AuriLab/gpt-bi-demo)

### **Utilizando Transformers**
Si prefieres experimentar con el modelo en un entorno de programación, te proporcionamos un cuaderno de Google Colab listo para usar <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">. Para utilizar el modelo puedes utilizar la librería `transformers` de Hugging Face.

#### **Instalación**
Primero, asegúrate de tener instalada la biblioteca `transformers` y `torch` (o `tensorflow`). Puedes instalarlas con pip:
```bash
pip install transformers torch
```
#### **Cargar el modelo y generar texto**
Una vez instalado `transformers` puedes utilizar `pipeline` para empezar a generar texto:
```python
from transformers import pipeline

pipe = pipeline("text-generation", model = "AuriLab/gpt-bi")

generated_text = pipe("Bazen behin", max_length=50, do_sample=True, temperature=0.7)

print(generated_text[-1]["generated_text"])
```

Como con cualquier otro modelo de `transformers` puedes ajustar los parámetros de generación, como max_length, temperature, top_k, y top_p, para controlar la creatividad y la longitud del texto generado.

## **Sobre el modelo**
GPT-Bi está basado en la arquitectura de GPT-2 (Radford et al., 2019), un modelo de lenguaje transformador ampliamente reconocido por su capacidad para generar texto coherente y contextualmente relevante. Este modelo cuenta con aproximadamente 110 millones de parámetros, lo que lo convierte en una opción ligera pero potente para tareas de generación de texto en euskera. El tamaño del vocabulario es de ~32,000 tokens, lo que permite una representación eficiente y precisa del idioma, capturando su riqueza léxica y gramatical.

En su estado actual, GPT-Bi es un modelo de lenguaje generalista, lo que significa que no está específicamente entrenado para seguir instrucciones o realizar tareas guiadas por el usuario. Sin embargo, está previsto desarrollar una versión "instruct" en el futuro, que permitirá una interacción más directa y orientada a tareas específicas, ampliando así su utilidad en aplicaciones prácticas. En cuanto a las características técnicas del modelo, tiene una longitud máxima de secuencia de 1,024 tokens. El modelo utiliza una dimensionalidad de embedding de 768, 12 capas de transformadores y 12 cabezas de atención.

## **Sobre el dataset**
El entrenamiento de GPT-Bi se ha basado en un conjunto de datos cuidadosamente seleccionado y procesado para garantizar un rendimiento óptimo en la generación de texto en euskera. Los tokens han sido contados utilizando el tokenizer propio de GPT-Bi.  

El conjunto de datos de entrenamiento consta de aproximadamente ~2,200 millones de tokens, ~1,900M en Euskera y ~300M en Castellano. Esta cantidad ha sido elegida para lograr un entrenamiento óptimo desde el punto de vista computacional (Hoffmann et al., 2022). Está compuesto por dos fuentes principales: una en euskera y otra en castellano. La inclusión del castellano se debe a la insuficiencia de tokens en euskera para realizar el preentrenamiento exclusivamente en este idioma.

### Corpus en Euskera
Se ha utilizado `HiTZ/latxa-corpus-v1.1` (Etxaniz et al., 2024). Este conjunto de datos ha sido recopilado y procesado por el equipo de HiTZ, combinando diversas fuentes existentes junto con nuevas incorporaciones. La selección de este dataset responde a su amplia cobertura del euskera y su calidad tras los procesos de deduplicación y limpieza.

A continuación, se detallan las fuentes de datos incluidas en [`HiTZ/latxa-corpus-v1.1`](https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1):
| Fuente             | Descripción |
|-------------------|-------------|
| **EusCrawl v1.1** | Versión actualizada de EusCrawl v1 (Artetxe et al., 2022), con contenido hasta noviembre de 2023. |
| **Egunkaria** | Contenido del diario Egunkaria. |
| **Booktegi** | Libros en formato EPUB provenientes de [Booktegi](https://www.booktegi.eus/). |
| **Wikipedia** | Dump de la Wikipedia en euskera correspondiente a noviembre de 2023 ([Wikimedia](https://huggingface.co/datasets/wikimedia/wikipedia)). |
| **CulturaX** | Porción en euskera del corpus CulturaX (Nguyen et al., 2023). |
| **Colossal OSCAR** | Porción en euskera de varias versiones del corpus [Colossal OSCAR](https://huggingface.co/datasets/oscar-corpus/colossal-oscar-1.0). |
| **HPLT v1** | Porción en euskera del corpus HPLT v1 (Aulamo et al., 2023). |

Además, la distribución de documentos en los conjuntos de entrenamiento, validación y prueba es la siguiente:
| Fuente             | Train       | Valid   | Test    |
|-------------------|------------|---------|---------|
| **CulturaX**     | 1,283,429  | 13,096  | 13,098  |
| **EusCrawl v1.1** | 1,758,084  | 17,861  | 17,736  |
| **HPLT v1**       | 367,238    | 3,797   | 3,699   |
| **Colossal OSCAR**| 233,753    | 2,483   | 2,276   |
| **Wikipedia**     | 400,902    | 4,063   | 4,092   |
| **Egunkaria**     | 172,876    | 1,766   | 1,764   |
| **Booktegi**      | 161        | 4       | 1       |

Para más detalles sobre las licencias y características de cada conjunto de datos, se recomienda consultar las referencias correspondientes en la publicación original del corpus [`HiTZ/latxa-corpus-v1.1`](https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1).

### Corpus en Castellano
En cuanto al corpus en castellano, se han extraido los ~300M de tokens en castellano necesarios para llegar al óptimo de tokens del dump de Wikipedia realizado por Wikimedia [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia).

## **Entrenamiento**
[PROXIMAMENTE]
```
git clone https://github.com/ErikSarriegui/gpt-bi

cd gpt-bi

pip install -r requirements.txt

torchrun --nproc_per_node=<n_gpus> train.py
```

## **Equipo**
[PROXIMAMENTE]

## **Agradecimientos**
[PROXIMAMENTE]

## **Apoya el proyecto**
[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=tuusuario&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/eriksarriegui)

## **Bibliografía**
* HOFFMANN, Jordan, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
* Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.
* Etxaniz, Julen, et al. "Latxa: An open language model and evaluation suite for Basque." arXiv preprint arXiv:2403.20266 (2024).
* Artetxe, Mikel, et al. "Does corpus quality really matter for low-resource languages?, 2022." URL: https://arxiv. org/abs/2203.08111. doi 10.
* Nguyen, Thuat, et al. "Culturax: A cleaned, enormous, and multilingual dataset for large language models in 167 languages." arXiv preprint arXiv:2309.09400 (2023).
* Aulamo, Mikko, et al. "HPLT: High performance language technologies." Annual Conference of The European Association for Machine Translation. European Association for Machine Translation, 2023.
