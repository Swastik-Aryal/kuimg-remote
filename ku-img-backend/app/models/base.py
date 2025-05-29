import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image	

class MultiClassSingleTagModel : 
    """ Base Class for Models which is trained on multiple tags and outputs a single tag """
    def __init__(self, model_path, tags) : 
        self.__model_path = model_path
        self.__tags = tags
        self.__model = self.import_model(model_path)


    def supported_tags(self) :
        """ Returns all the tags supported by the current model """
        return self.__tags


    def model(self) : 
        """ Returns the underlying model """
        return self.__model


    def model_path(self) : 
        """ Returns the path that contains the underlying model """
        return self.__model_path


    def import_model(self, model_path) : 
        """
            This method implements the logic necessary for importing the exported module dynamically.
            Returns : a python instance
        """
        raise NotImplementedError('import_model method needs to be implemented')


    def transform_input(self, img_path) : 
        """
            This method specifies the logic necessary to transform the input image into an input format supported by the model
            Retuns : an array/object that can be directly input to the model.
        """
        raise NotImplementedError('transform_input method needs to be implemented')


    def predict(self, model_input) : 
        """
            This method specifies the logic to get the output of the ML model.
            Returns : an array/object that can be parsed to derive tag
        """
        raise NotImplementedError('predict method has not been implemented')


    def transform_output(self, model_output) : 
        """
            This method specifies the logic necessary to transform the output of the model into a tag.
            Returns : a string that represents a tag. This must be present in self.__tags or None.

            One critical note here is to ensure that None is properly returned.
            if a model does not have confidence in any particular tag, it should return a None. 
            Any string/tag that is returned is interpretted as a successful tag, which might result in output confusion.

        """
        raise NotImplementedError('transform_output method has not been implemented')


    def tag(self, img) : 
        """
            This method returns a single tag (str) as output
        """
        transformed_img = self.transform_input(img)
        model_prediction = self.predict(transformed_img)
        tag = self.transform_output(model_prediction)
        return tag


class MultiClassSingleTagKerasStandardModel(MultiClassSingleTagModel) : 
    """
        Base for all Keras Model which adheres to MultiClassSingleTag Model.
    """
    def __init__(self, model_path, tags) : 
        super().__init__(model_path, tags)

    def import_model(self, model_path) : 
        return keras.models.load_model(model_path)

    def predict(self, model_input) : 
        return self.model().predict(model_input)
