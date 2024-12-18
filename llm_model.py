import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from huggingface_hub import snapshot_download
from typing import Tuple
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")

class TextModel:
    #@ensure_annotations
    def __init__(self, model_name:str, model_dir:str,max_tokens:int = 1024, device: str = "cuda", temperature:float = 0.2, top_p:float = 0.6,\
                 top_k:int = None, num_return_seq:int = 2, rep_penalty:float = 2.5, do_sample:bool = True, \
                 ):
        """
        Intialized the model and sets generation configuration
        Args:
            model_file (str): Model file path in string format
            max_tokens (int): Maximum number of new tokens to generate
            temperature (int): Sampling temperature; min: 0.2 max: 1.0
            top_p (float): Nucleus sampling probablity; min: 0.3 max:0.9
            top_k (int): Top-K sampling, min: 10 max: 100
            num_return_seq (int): Number of sequences to return; min:1 max: 5 default:1
            rep_penality (float): Repetition penalty, min: 1 max:3, default:3.0
            do_sample (bool): Whether to use sampling.
        Raises:
            TypeError: If the model file is not a string. 
        """
        #self.save_dir = os.path.join(os.getcwd(),model_dir)
        logger.info(f"Device: {device}")
        self.device = device
        #User-defined generation configuration
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_return_seq = num_return_seq
        self.rep_penalty = rep_penalty
        self.do_sample = do_sample
        #self.model_name = model_name
        self.tokenizer, self.model = self._download_and_load_model(model_name=model_name,save_dir=model_dir)
        #Load and configure the generation config
        #self.generation_config = GenerationConfig()
        self.generation_config = self._load_default_config() # Applying default or user configuration
      
    
    def _load_default_config(self)-> None:
        """
        Load generation configuration settings based on user-defined or default values.
        """
        """  self.generation_config.max_new_tokens = self.max_tokens
        self.generation_config.temperature = self.temperature
        self.generation_config.top_p = self.top_p
        self.generation_config.top_k = self.top_k
        self.generation_config.num_return_sequences = self.num_return_seq
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.repetition_penalty = self.rep_penalty
        self.generation_config.do_sample = self.do_sample"""

        try:
            # Create generation config
            generation_config = GenerationConfig()
            # Intelligent token ID selection
            generation_config.pad_token_id = (
                self.tokenizer.pad_token_id or 
                self.tokenizer.eos_token_id or 
                self.tokenizer.bos_token_id
            )
            generation_config.eos_token_id = self.tokenizer.eos_token_id
            generation_config.do_sample = self.do_sample
            if self.do_sample:
                # Rest of your configuration
                generation_config.top_p = self.top_p
                generation_config.top_k = self.top_k
            generation_config.repetition_penalty = self.rep_penalty
            generation_config.num_return_sequences = self.num_return_seq
            generation_config.max_new_tokens = self.max_tokens
            generation_config.temperature = self.temperature
            # ... other configurations
            return generation_config
        except Exception as e:
            logger.error(f"Error in generation config setup: {e}")
            raise
    
    def _download_and_load_model(self, model_name: str, save_dir: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """downloading both tokenizer and model if doesn't exist locally"""
        if model_name is None:
            logger.error(f"[Error] Model name is empty")
            raise ValueError(f"[Error] Model name is empty")
        save_path = os.path.join(os.getcwd(), save_dir)
        if not os.path.exists(save_path):
            logger.error(f"{save_dir} folder not found")
            #raise FileNotFoundError(f"Model file {saved_model} not found")
            self._download_snapshot(model_name= model_name, model_save_path=save_path)

        tokenizer, model = self._load_tokenizer_and_model(model_name= model_name, model_dir=save_path)
        return tokenizer, model

    def _load_tokenizer_and_model(self, model_name:str, model_dir:str ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Loads the both tokenizer and text generation model

        Returns:
            AutoTokenizer: The loaded tokenizer instance.
            AutoModelForCausalLM: The loaded model instance.

        Raises: 
            IOError: If there is an error loading the tokenizer and text model due to model path issues.
        """
        try:
            logger.info(f"Loading tokenizer from '{os.path.basename(model_dir)}' folder.") 
            tokenizer = AutoTokenizer.from_pretrained(model_dir, torch_dtype='auto', trust_remote =True)
            tokenizer.pad_token = tokenizer.eos_token 
        except IOError as e:
            logger.error(f"[Error] Unable to load tokenizer from {os.path.basename(model_dir)}")
            raise IOError(f"Error loading Tokenizer from {os.path.basename(model_dir)}: {e} ")
            #tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=model_dir)
            #tokenizer.pad_token = self.tokenizer.eos_token
            #logger.info(f"Downloaded and loaded tokenizer from {model_dir} folder")
        try:
            logger.info(f"Loading model from '{os.path.basename(model_dir)}' folder")  
            model = AutoModelForCausalLM.from_pretrained(model_dir,revision="main", torch_dtype='auto',trust_remote_code=True)
            model = model.to(self.device)
            logger.info(f"Model loaded successfully..!!!")
        except IOError as e:
            logger.error(f"[Error] Unable to load {model_name} from {os.path.basename(model_dir)}")
            raise IOError(f"Error loading {model_name} from {os.path.basename(model_dir)}: {e} ")
            #self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.save_dir, device_map = "auto")
            #logger.info(f"Model downloaded and loaded from {self.save_dir} folder")
        return tokenizer, model
    
    def _download_snapshot(self, model_name, model_save_path):
        #model_name = "sentence-transformers/all-mpnet-base-v2"
        try:
            #if not os.path.exists(model_save_path):
            logger.info(f"Creating {os.path.basename(model_save_path)} folder")
            os.makedirs(model_save_path)
            logger.info(f"Downloading '{model_name}' into '{os.path.basename(model_save_path)}'")  
            snapshot_download(repo_id=model_name,local_dir=model_save_path)
        except Exception as e:
            logger.error(f"Error while downloading '{model_name}' into '{model_save_path}'")
            raise Exception(f"Error while downloading '{model_name}' into '{model_save_path}': {e}") 

    def model_response(self,message: str, skip_special_tokens: bool = True) -> str:
        """
        Args:
            message(str): Input message in string format
        Returns:
            str: Model generated response in string format 
        Raises:
            TypeError: If the input message is not a string 
        """
        #if not isinstance(message, str):
        #    raise TypeError(f"Input message is not a string...!!!")
        encoding = self.tokenizer(message.strip(), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids = encoding.input_ids,
                attention_mask = encoding.attention_mask,
                generation_config = self.generation_config)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
        return response
