"""
This module contains a zero-order representation of a photovoltaic
operation.
"""

from idaes.core import declare_process_block_class

from watertap.core import build_pt, ZeroOrderBaseData
from watertap_contrib.seto.energy import solar_energy

__author__ = "Kurban Sitterley"


@declare_process_block_class("PhotovoltaicZO")
class PhotovoltaicZOData(ZeroOrderBaseData):
    """
    Zero-Order model for photovoltaic system.
    """

    def build(self):
        super().build()

        self._tech_type = "photovoltaic"

        build_pt(self)
        solar_energy(self)
