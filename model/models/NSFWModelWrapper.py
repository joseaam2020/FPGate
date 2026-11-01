from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Dict

@dataclass
class Prediction:
    filepath: str
    model_name: str
    nsfw_probability: float
    predicted_label: str
    latency_ms: float
    raw_output: str = "" # Para texto completo de LlavaGuard

class NSFWModelWrapper(ABC):
    """
    Interface para que el harness pueda correr todos los modelos de la misma manera
    """

    name: str = "unamed_model"
    current_filepath: str = "empty" 
    image = None

    @abstractmethod
    def _predict_probability(self) -> float:
        """
        Retorna la probabilidad de NSFW (0-1) para una imagen y guarda el filepath si necesario
        
        Returns:
            Probabilidad de que la imagen sea nsfw (float de 0-1)
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, filepath: str) -> None:
        """
        Ejecuta cualquier pre-procesamiento necesario para la imagen en cuestion

        Args:
            filepath: direccion a la imagen a procesar
        """
        raise NotImplementedError

    def predict(self, filepath: str, threshold: float = 0.5) -> Prediction:
        """
        Wrapper con manejo de tiempo y errores para procesar una sola imagen

        Args:
            filepath: direccion de la imagen a procesar
            threshold: valor de clasificacion binaria nsfw

        Returns:
            Resultado de la prediccion (Prediction)
        """
        self._preprocess(filepath=filepath) 

        start = perf_counter()
        probability = self._predict_probability()
        elapsed_ms = (perf_counter() - start) * 1000

        probability = max(0.0, min(1.0, float(probability))) # Normalizar el valor
        label = "NSFW" if probability >= threshold else "SFW"

        return Prediction(
            filepath=filepath,
            model_name=self.name,
            nsfw_probability=probability,
            predicted_label=label,
            latency_ms=elapsed_ms
        )

    def predict_batch(self, filepaths: List[str], threshold: float = 0.5) -> Dict[str, Prediction]:
        """
        Wrapper con manejo de tiempo y errores para procesar una sola imagen

        Args:
            filepaths: lista con las direcciones de las imagenes a procesar
            threshold: valor de clasificacion binaria nsfw

        Returns:
            Diccionario de la forma direccion: prediccion
        """
        predictions = {}
        for filepath in filepaths:        
            predictions[filepath] = self.predict(filepath=filepath, threshold=threshold)

        return predictions


        
        
    