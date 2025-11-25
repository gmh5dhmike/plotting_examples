import numpy as np
import matplotlib.pyplot as plt


def make_canvas1():
    """
    Canvas 1: function plot (Matplotlib version of ROOT canvas1).
    Adjust to match the ROOT example more closely if needed.
    """
    x = np.linspace(0, 4 * np.pi, 1000)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y1, label="sin(x)", linewidth=2)
    plt.plot(x, y2, label="cos(x)", linewidth=2)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Canvas 1 (Matplotlib)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("canvas1_py.png", dpi=150)
    plt.close()


def make_canvas2():
    """
    Canvas 2: histogram + Gaussian overlay (Matplotlib version of ROOT canvas2).
    """
    N = 10000
    mu = 100
    sigma = 10

    data = np.random.normal(mu, sigma, N)

    nbins = 50
    counts, bin_edges = np.histogram(data, bins=nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    errors = np.sqrt(counts)

    plt.figure(figsize=(8, 6))

    plt.step(bin_edges[:-1], counts, where='post', label="Data (counts)", linewidth=1.5)
    plt.errorbar(
        bin_centers,
        counts,
        yerr=errors,
        fmt='o',
        markersize=3,
        capsize=2,
        label="Stat. errors (√N)"
    )

    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    pdf_scaled = pdf * N * bin_width
    plt.plot(x, pdf_scaled, 'r-', linewidth=2, label="Gaussian PDF (scaled)")

    plt.xlabel("Value")
    plt.ylabel("Counts per bin")
    plt.title("Canvas 2 (Matplotlib)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("canvas2_py.pdf")
    plt.close()


def main():
    make_canvas1()
    make_canvas2()
    print("Created canvas1_py.png and canvas2_py.pdf using Matplotlib.")


if __name__ == "__main__":
    main()
