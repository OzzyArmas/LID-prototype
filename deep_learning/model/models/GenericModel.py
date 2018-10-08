class Model:
    def train(self, x, y):
        '''
        train model on feature set x with labels y
        '''
        pass

    def predict(self, x):
        '''
        return predicted label
        '''
        pass
    def predict_all(self, x):
        '''
        predict all labels
        '''
        pass
    def adjust_data_to_model(self,data):
        '''
        return adjusted data to fit shape of model
        '''
        pass
def ClassName():
    return Model