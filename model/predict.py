from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense

import cv2
import shutil
import numpy as np
import os
import joblib
import time


class VideoDescription:

    def __init__(self, video_name, file_path, feat_path='model/temps', save_model_path='model/trained_models'):
        self.video_name = video_name
        self.file_path = file_path
        self.feat_path = feat_path
        self.save_model_path = save_model_path

        self.latent_dim = 512  # The number of hidden features for lstm
        self.num_decoder_tokens = 1500  # The number of features from each frame

        self.tokenizer, self.inv_map = self.load_tokenizer()
        self.inf_encoder_model = self.load_encoder()
        self.inf_decoder_model = self.load_decoder()

        if not os.path.isdir(os.path.join(self.feat_path, 'feat')):
            os.mkdir(os.path.join(self.feat_path, 'feat'))

    def vgg_load(self):
        model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)
        return model_final

    def video_to_frames(self, video):
        path = os.path.join(self.feat_path, 'temporary_images')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        video_path = os.path.join(self.file_path, video)
        count = 0
        image_list = []

        # Path to video file
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            cv2.imwrite(os.path.join(self.feat_path, 'temporary_images', 'frame%d.jpg' % count), frame)
            image_list.append(os.path.join(self.feat_path, 'temporary_images', 'frame%d.jpg' % count))
            count += 1

        cap.release()
        cv2.destroyAllWindows()
        return image_list

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        return img

    def extract_features(self, video, model):
        """
        :param video: The video whose frames are to be extracted to convert into a numpy array
        :param model: the pretrained vgg16 model
        :return: numpy array of size 4096x80
        """
        video_id = video.split(".")[0]
        print(video_id)
        print(f'Processing video {video}')

        image_list = self.video_to_frames(video)
        samples = np.round(np.linspace(
            0, len(image_list) - 1, 80))
        image_list = [image_list[int(sample)] for sample in samples]
        images = np.zeros((len(image_list), 224, 224, 3))
        for i in range(len(image_list)):
            img = self.load_image(image_list[i])
            images[i] = img
        images = np.array(images)

        fc_feats = model.predict(images, batch_size=128)
        img_feats = np.array(fc_feats)

        # cleanup
        shutil.rmtree(os.path.join(self.feat_path, 'temporary_images'))
        return img_feats

    def load_tokenizer(self):
        with open(os.path.join(self.save_model_path, 'tokenizer' + str(self.num_decoder_tokens)), 'rb') as file:
            tokenizer = joblib.load(file)
        # inverts word tokenizer
        inv_map = {value: key for key, value in tokenizer.word_index.items()}

        return tokenizer, inv_map

    def load_encoder(self):
        # loading encoder model. This remains the same
        return load_model(os.path.join(self.save_model_path, 'encoder_model.h5'))

    def load_decoder(self):
        # inference decoder model loading
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        inf_decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        inf_decoder_model.load_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))

        return inf_decoder_model

    def greedy_search(self, loaded_array):
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, 1500))
        sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        for i in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)
            if y_hat == 0:
                continue
            if self.inv_map[y_hat] is None:
                break
            else:
                sentence = sentence + self.inv_map[y_hat] + ' '
                target_seq = np.zeros((1, 1, 1500))
                target_seq[0, 0, y_hat] = 1
        return ' '.join(sentence.split()[:-1])

    def extract_and_save_features(self):
        outfile = os.path.join(self.feat_path, 'feat', self.video_name + '.npy')
        vgg_model = self.vgg_load()
        img_feats = self.extract_features(self.video_name, vgg_model)
        np.save(outfile, img_feats)

    def predict(self):
        x_test = np.load(os.path.join(self.feat_path, 'feat', self.video_name + '.npy'))
        start = time.time()
        decoded_sentence = self.greedy_search(x_test.reshape(-1, 80, 4096))
        return decoded_sentence, ',{:.2f}'.format(time.time() - start)

    def extract_features_and_predict(self):
        self.extract_and_save_features()
        return self.predict()
