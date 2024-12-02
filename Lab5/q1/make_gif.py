import os
import imageio

def make_directory_structure():
    os.makedirs('./animation', exist_ok=True)

def make_gif(name):
    png_dir = f'images/{name}'
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    for _ in range(10):
        images.append(imageio.imread(file_path))

    imageio.mimsave(f'animation/{name}.gif', images)

if __name__ == '__main__':
    make_directory_structure()
    make_gif('vanilla')
    make_gif('average')