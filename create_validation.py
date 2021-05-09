from os.path import join, exists
from os import listdir, makedirs
from shutil import move
import random

# TODO: update genus name here
species = [
    "Alopias",
    "Asymbolus",
    "Carcharhinus",
    "Carcharias",
    "Carcharodon",
    "Cephaloscyllium",
    "Cetorhinus",
    "Chiloscyllium",
    "Chimaera",
    "Echinorhinus",
    "Etmopterus",
    "Eucrossorhinus",
    "Galeocerdo",
    "Galeorhinus",
    "Galeus",
    "Ginglymostoma",
    "Haploblepharus",
    "Hemipristis",
    "Hemiscyllium",
    "Heterodontus",
    "Hexanchus",
    "Hydrolagus",
    "Isurus",
    "Lamna",
    "Megachasma",
    "Mustelus",
    "Nebrius",
    "Negaprion",
    "Notorynchus",
    "Orectolobus",
    "Poroderma",
    "Prionace",
    "Rhincodon",
    "Scyliorhinus",
    "Sphyrna",
    "Squalus",
    "Squatina",
    "Stegostoma",
    "Triaenodon",
    "Triakis"
]

# TODO: update train and validation folder
train_dir = "train/"
validation_dir = "validation/"


def create_validation():
    """Validation data sepration from augmented training images.
    Number of images chosen for validation depends upon the
    number of images present in the directory. If less than 78,
    then 6 images are moved into validation folder. Similarly,
    two if conditions for cases with less than 81 and greater
    than 85. Images are selected using random sampling.
    """
    for bird_specie in species:
        destination = ""
        train_imgs_path = join(train_dir, bird_specie)
        if not exists(join(validation_dir, bird_specie)):
            destination = makedirs(join(validation_dir, bird_specie))

        else:
            train_imgs = listdir(train_imgs_path)
            number = len(train_imgs)  # number of images in each category
            if number < 78:
                validation_separation = random.sample(train_imgs, 6)
                for img_file in validation_separation:
                    move(join(train_imgs_path, img_file), join(destination, img_file))

            elif 78 <= number <= 81:
                validation_separation = random.sample(train_imgs, 10)
                for img_file in validation_separation:
                    move(join(train_imgs_path, img_file), join(destination, img_file))

            elif number > 85:
                validation_separation = random.sample(train_imgs, 20)
                for img_file in validation_separation:
                    move(join(train_imgs_path, img_file), join(destination, img_file))


if __name__ == "__main__":

    create_validation()
