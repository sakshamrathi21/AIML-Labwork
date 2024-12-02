import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []
    

    # TODO: first generate random numbers from the uniform distribution
    if (distribution == "exponential"):
        uniform_samples = np.random.rand(num_samples)
        inverse_exponential_cdf = lambda x: -np.log(1-x)/kwargs.get("lambda")
        for i in range(num_samples):
            x = inverse_exponential_cdf(uniform_samples[i])
            samples.append(round(x, 4))
        # unrounded_samples = inverse_exponential_cdf(uniform_samples)
        # samples = np.round(unrounded_samples, decimals=4)

    elif (distribution == "cauchy"):
        uniform_samples = np.random.rand(num_samples)
        inverse_cauchy_cdf = lambda x: np.tan(np.pi*(x-0.5))*kwargs.get("gamma") + kwargs.get("peak_x")
        # z1 = np.random.rand(num_samples)
        # z2 = np.random.rand(num_samples)
        for i in range(num_samples):
            x = inverse_cauchy_cdf(uniform_samples[i])
            samples.append(round(x, 4))

    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples)
        plt.savefig("q1_" + distribution + ".png")
        # plt.show()
        # END TODO
