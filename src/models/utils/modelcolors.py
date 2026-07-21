model_colors = {
    # baseline:
    'constantmodel'     : "#B0B0B0",
    'persistencemodel'  : "#888888",
    'climateologymodel' : "#595959",
    'climascalemodel'   : "#2E2E2E",

    # shallow
    'noderfmodel'       : '#35B714',
    'shallow2'          : '#5BE838',
    'shallow3'          : '#9AF184',
    'shallow4'          : '#D9F9D1',

    # deep:
    #   vanilla
    'nodelstmmodel'     : '#1467B7',
    'lstmmodel'         : '#1467B7',
    'seqnodelstmmodel'  : '#3892E8',
    'nodebilstmmodel'   : '#84BCF1',
    'nodegrumodel'      : '#D1E5F9',
    #   gnn
    'gcnmodel'         : '#B77914',
    'gatmodel'          : "#14B78632"
    
}
from matplotlib.colors import to_rgb
def color_is_light(color: str, threshold=0.6):
    # perceived luminance (human-vision–weighted)
    r, g, b = to_rgb(color)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return luminance > threshold
