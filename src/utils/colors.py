import seaborn as sns
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib

blackcolor = (0,0,0)
traincolor = '#4a90d9'
valcolor   = "#1b9e77"
testcolor  = '#d94e4e'  


large_pallete_blue  = sns.color_palette("Blues", n_colors=100)
large_pallete_red   = sns.color_palette("Reds",  n_colors=100)
paired_colors       = sns.color_palette("Paired")

models_cmap = mypalette = ["#5ab1e0",'#1f78b4','#1b9e77','#33a02c','#2a7d32',"#9154ac",'#6a3d9a','#ffb84d','#a6761d','#b15928','#8b3d3d','#666666',"#b1adad",'#000000']

def inspect_colorpalette(palette_name: Optional[str] = None, n_colors: Optional[int] = 10, palette_list: Optional[List[str]]=None, get_colors: bool = True):
    """
    Inspect a colorpalette, albeit existing (then use palette_name) or manually created (palette_list)

    Parameters
    ----------
    palette_name: Optional[str] = None
        the name with under which the palette can be found in sns.color_palette
    n_colors: Optional[int] = 10
        the number of colors to show
    palette_list: Optional[List[str]] = None
        a manually defined list of colors (hex_codes preferred)
    get_colors: bool = True
        whether or not to return the list of colors

    Returns
    -------
    Figure of the colorpalette

    Printed text of the colors

    list of hex codes, returned if get_colors

    Examples
    --------
    >>> inspect_colorpalette('model colors', palette_list = models_cmap)
    >>> inspect_colorpalette('Paired', n_colors = 12)
    """

    # if manually_defined palette:
    if palette_list:
        palette = palette_list

        name = palette_name if palette_name else 'unnamed palette'    

    else:
        if palette_name:
            try:
                # Load the palette
                palette = sns.color_palette(palette_name, n_colors=n_colors)
                name    = palette_name
            except ValueError:
                print(f"Error: '{palette_name}' is not a valid palette name.")
                return
        else:
            raise ValueError('Please supply either a palette_name or a palette_list')            
        
        
        
    plt.figure(figsize=(10, 2))        
    sns.palplot(palette)

    p_list = []
    for i, color in enumerate(palette):
            hex_code = matplotlib.colors.to_hex(color)
            p_list.append(hex_code)
            # Display number and hex code underneath the color
            plt.text(i, -0.1, f"{i}", ha='center', va='top', color='black', fontsize=12)    

    plt.title(name)
    plt.show()

    print(f"Color codes for the '{name}' palette:")
    for i, color in enumerate(palette):
        hex_code = matplotlib.colors.to_hex(color)
        print(f"{i}: {hex_code}")    

    if get_colors:
        return p_list