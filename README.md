# REQUIREMENTS:
- numpy
- scipy
- opencv-python
- matplotlib
- python-dev
- pydicom
- NURBS-Python
- plotly
- anaconda
- orca
- psutil

# INSTALATION ON UBUNTU:
- sudo apt-get update
- sudo apt-get upgrade

- pip:
  sudo apt-get install python-pip

- numpy and scipy:
  sudo pip install numpy scipy

- opencv-python:
  sudo apt-get install python-opencv

- matplotlib:
  sudo pip install matplotlib

- pydicom:
pip install -U pydicom or sudo pip install pydicom

 - If an error was found, read: https://pydicom.github.io/pydicom/stable/transition_to_pydicom1.htmls

 - User guide: https://pydicom.github.io/pydicom/stable/pydicom_user_guide.html

- plotly:
  sudo pip install plotly

- NURBS-Python
  - Documentation:
    https://nurbs-python.readthedocs.io/en/latest/
  - NURBS-Python Github:
    https://github.com/orbingol/NURBS-Python
  - NURBS-Python Examples:
    https://github.com/orbingol/NURBS-Python_Examples

- anaconda:
https://conda.io/docs/user-guide/install/linux.html

- orca and psutil
conda install -c plotly-orca psutil

# EXECUTION SCRIPTS:

- Mapping the available sequences:

python3 mapping.py -patient=Iwasawa -mask=1

- Create 2DST images, using the original images in RM and using the silhouetes of the lungs:

python create2DST.py -patient=Iwasawa -mask=0

- Extract diaphragmatic respiratory function:

python respiratory_pattern.py -patient=Iwasawa -plan=Sagittal

- Smooth respiratory pattern (diaphragmatic respiratory function):

python smooth_respiratory_pattern.py -patient=Iwasawa -plan=Sagittal

- Register and reconstruction the diaphragmatic surface using Abe (2013) method:

python register.py -mode=2 -side=0 -imgnumber=9

- Manual segmentation:

python segmentation.py -plan=Coronal -sequence=8 -side=1 -show=1 -save=0 -imgnumber=1

- Register and plot the silhouete points of the lungs:

python3 register.py -mode=0 -patient=Iwasawa -rootsequence=9 -side=0 -imgnumber=4

- Spatio-temporal point interpolation in positions where there are no registers

python3 spatiotemporal_interpolation.py -rootsequence=9 -side=0 -imgnumber=9

- B-spline point interpolation

python3 spline_interpolation.py -rootsequence=9 -side=0 -imgnumber=1

- (Extra) Diaphragm reconstruction (Abe, 2013):

python3 diaphragm.py -mode=1 -patient=Iwasawa -rootsequence=9 -side=2 -imgnumber=1
