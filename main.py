import preprocessing
import model
import train
import generate

if __name__ == '__main__':
    # Preprocess the audio files
    audio, mfccs, shifted_audio = preprocessing.preprocess_audio('path/to/audio/recording.mp3')

    # Define the GAN model
    generator = model.make_generator_model()
    discriminator = model.make_discriminator_model()

    # Train the GAN model
    train.train(generator, discriminator, audio, mfccs, shifted_audio)

    # Generate new audio using the trained GAN model
    generated_audio = generate.generate_audio(generator, tf.random.normal([1, 100]))

    # Save the generated audio
    librosa.output.write_wav('generated_audio.wav', generated_audio[0], sr)
