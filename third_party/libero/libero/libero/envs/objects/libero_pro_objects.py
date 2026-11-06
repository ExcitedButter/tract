"""LIBERO-Pro BDDL aliases: map variant type names to existing MuJoCo assets."""

from libero.libero.envs.base_object import register_object
from libero.libero.envs.objects.articulated_objects import FlatStove, WhiteCabinet
from libero.libero.envs.objects.google_scanned_objects import (
    AkitaBlackBowl,
    Basket,
    Plate,
)
from libero.libero.envs.objects.hope_objects import (
    AlphabetSoup,
    BbqSauce,
    Butter,
    ChocolatePudding,
    CreamCheese,
    Ketchup,
    Milk,
    SaladDressing,
    TomatoSauce,
)
from libero.libero.envs.objects.turbosquid_objects import MokaPot, WineBottle, WineRack


@register_object
class BiggerAlphabetSoup(AlphabetSoup):
    def __init__(self, name="bigger_alphabet_soup", obj_name="alphabet_soup"):
        super().__init__(name, obj_name)


@register_object
class BiggerAkitaBlackBowl(AkitaBlackBowl):
    def __init__(self, name="bigger_akita_black_bowl", obj_name="akita_black_bowl"):
        super().__init__(name, obj_name)


@register_object
class RedCreamCheese(CreamCheese):
    def __init__(self, name="red_cream_cheese", obj_name="cream_cheese"):
        super().__init__(name, obj_name)


@register_object
class BlueKetchup(Ketchup):
    def __init__(self, name="blue_ketchup", obj_name="ketchup"):
        super().__init__(name, obj_name)


@register_object
class YellowTomatoSauce(TomatoSauce):
    def __init__(self, name="yellow_tomato_sauce", obj_name="tomato_sauce"):
        super().__init__(name, obj_name)


@register_object
class GreenBbqSauce(BbqSauce):
    def __init__(self, name="green_bbq_sauce", obj_name="bbq_sauce"):
        super().__init__(name, obj_name)


@register_object
class GreenButter(Butter):
    def __init__(self, name="green_butter", obj_name="butter"):
        super().__init__(name, obj_name)


@register_object
class GreenChocolatePudding(ChocolatePudding):
    def __init__(self, name="green_chocolate_pudding", obj_name="chocolate_pudding"):
        super().__init__(name, obj_name)


@register_object
class YellowMilk(Milk):
    def __init__(self, name="yellow_milk", obj_name="milk"):
        super().__init__(name, obj_name)


@register_object
class RedSaladDressing(SaladDressing):
    def __init__(self, name="red_salad_dressing", obj_name="salad_dressing"):
        super().__init__(name, obj_name)


@register_object
class RedBasket(Basket):
    def __init__(self, name="red_basket", obj_name="basket"):
        super().__init__(name, obj_name)


@register_object
class BlackBowl(AkitaBlackBowl):
    def __init__(self, name="black_bowl", obj_name="akita_black_bowl"):
        super().__init__(name, obj_name)


@register_object
class YellowPlate(Plate):
    def __init__(self, name="yellow_plate", obj_name="plate"):
        super().__init__(name, obj_name)


@register_object
class YellowMokaPot(MokaPot):
    def __init__(self, name="yellow_moka_pot", obj_name="moka_pot"):
        super().__init__(name, obj_name)


@register_object
class WhiteBottle(WineBottle):
    def __init__(
        self,
        name="white_bottle",
        obj_name="wine_bottle",
        joints=None,
    ):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)


@register_object
class YellowStove(FlatStove):
    """Same asset as FlatStove; fixtures are constructed with ``joints=None``."""

    def __init__(
        self,
        name="yellow_stove",
        obj_name="flat_stove",
        joints=None,
    ):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)


@register_object
class YellowCabinet(WhiteCabinet):
    """LIBERO-Pro alias used by goal-object tasks."""

    def __init__(
        self,
        name="yellow_cabinet",
        obj_name="white_cabinet",
        joints=None,
    ):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)


@register_object
class BrownRack(WineRack):
    """LIBERO-Pro alias for recolored wine rack in goal-object tasks."""

    def __init__(
        self,
        name="brown_rack",
        obj_name="wine_rack",
        joints=None,
    ):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        super().__init__(name=name, obj_name=obj_name, joints=joints)
