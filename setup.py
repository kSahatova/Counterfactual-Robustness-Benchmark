from setuptools import setup
  
setup( 
    name='src', 
    version='0.1', 
    description='An adapted CARLA package for recourse methods to generate counterfactual explanations for images', 
    author='Kseniya Sahatova', 
    author_email='sahatova.kseniya@gmail.com', 
    packages=["src", "src.datasets", "src.utils", "src.evaluation",
              "src.models", "src.trainers", "src.recourse_methods"], 
    include_package_data=True 
) 