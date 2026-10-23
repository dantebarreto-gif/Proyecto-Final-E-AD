"""Permite ejecutar los scripts sin instalar el paquete (agrega src/ al path)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
