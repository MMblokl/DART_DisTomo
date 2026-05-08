from source.phantoms.phantom_generators import gen_blob, gen_bone, gen_mesh

def generate_phantoms(seed: int = 202667, quantity: int = 5, save_location: str = "./phantoms", resolution: int = 512):
    """Generates all phantoms based on inputs."""
    gen_blob(seed, quantity, f"{save_location}/blobs/", resolution, 25, 75, 60, 10)
    gen_mesh(seed, quantity, f"{save_location}/meshes/", resolution, 100, 40)
    gen_bone(seed, quantity, f"{save_location}/bones/", resolution, 120, 60, 40, 80, 150, 220, 110)


if __name__ == "__main__":
    generate_phantoms()