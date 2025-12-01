"""Setup script for Brooklyn Crosswalk QA package."""

from setuptools import setup, find_packages

setup(
    name="crosswalk-qa-brooklyn",
    version="0.1.0",
    description="Crosswalk detection QA system for Brooklyn, NYC",
    author="CrossCheck-NYC",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "geopandas>=0.13.0",
        "shapely>=2.0.0",
        "rasterio>=1.3.0",
        "pyproj>=3.5.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "viz": [
            "streamlit>=1.22.0",
            "folium>=0.14.0",
            "pydeck>=0.8.0",
            "altair>=5.0.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crosswalk-qa=run_pipeline:main",
        ],
    },
)