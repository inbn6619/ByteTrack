import numpy as np
import cv2


class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (4,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (4,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape==(4,2), "Need (4,2) input array"
        assert lonlat_array.shape==(4,2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        lonlat = np.dot(self.M,pixel.T)
        
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T





# ###

# ch1 = np.array([
#     [0, 660],
#     [1819, 776],
#     [1330, 200],
#     [486, 162]
# ])

# ch2 = np.array([
# 	[24,950],
# 	[1832,917],
# 	[1334,347],
# 	[534,351],
# ])

# ch3 = np.array([
#     [11,824],
# 	[1832,757],
# 	[1415,98],
# 	[504,151],
# ])

###

###

ch1_mini = np.array([
	[38,648],
	[1280,648],
	[1280,72],
	[38,72]
])

ch2_mini = np.array([
	[1280,648],
	[2560,648],
	[2560,72],
	[1280,72],
])

ch3_mini = np.array([
    [2560,648],
	[3800,648],
	[3800,72],
	[2560,72],
])

###

### test


ch2t = np.array([
[1280, 0],
[1280, 0],
[1280, 0],
[1280, 0],
])

ch3t = np.array([
[2560, 0],
[2560, 0],
[2560, 0],
[2560, 0],
])

ch1 = np.array([
[9,501],
[1218,514],
[923,182],
[328,159],
])

ch2 = np.array([
[38,608],
[1211,589],
[922,270],
[323,281],
])

ch2 = np.array([
[38,608],
[1211,589],
[922,270],
[323,281],
])

ch3 = np.array([
[18,536],
[1225,465],
[885,124],
[291,163],
])


# ch1 = np.array([
#     [5,397],
#     [1245,423],
#     [925,128],
#     [323,111],
# ])

# ch1_mini = np.array([
# [101,481],
# [1280,481],
# [1280,33],
# [101,33],
# ])

# ch1_mini = np.array([
# [101,581],
# [1280,581],
# [1280,133],
# [101,133],
# ])

###


pm_1 = PixelMapper(ch1, ch1_mini)

pm_2 = PixelMapper(ch2 + ch2t, ch2_mini)

pm_3 = PixelMapper(ch3 + ch3t, ch3_mini)






# print('test')


# vid_720p = np.array([
#     [0, 720],
#     [1280, 720],
#     [1280, 0],
#     [0, 0],
# ])

# vid_1080p = np.array([
#     [0, 1080],
#     [1920, 1080],
#     [1920, 0],
#     [0, 0],
# ])

# change_720p_to_1080p = PixelMapper(vid_720p, vid_1080p)

# ch1 = change_720p_to_1080p.lonlat_to_pixel(ch1)
# ch2 = change_720p_to_1080p.lonlat_to_pixel(ch2)
# ch3 = change_720p_to_1080p.lonlat_to_pixel(ch3)


# area = np.array([
#         [5, 854],
#         [1902, 880],
#         [1386, 278], 
#         [491, 275]
#         ])


# minimap = np.array([
#     [0, 720],
#     [1280, 720],
#     [1280, 0],
#     [0,0],
# ])

# pm1 = PixelMapper(area, minimap)

