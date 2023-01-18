import tensorflow as tf

# Load the trained GAN model
generator = make_generator_model()
generator.load_weights('path/to/generator/weights')

# Generate new audio that sounds like the singer's voice
def generate_audio(model, input_noise):
    generated_audio = model(input_noise, training=False)
    return generated_audio

# Generate audio for a number of samples
for i in range(10):
    generated_audio = generate_audio(generator, tf.random.normal([1, 100]))
    librosa.output.write_wav('generated_audio_{}.wav'.format(i), generated_audio[0], sr)
