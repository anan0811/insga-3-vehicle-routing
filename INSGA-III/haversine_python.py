from numpy import sin, cos, arccos, pi, round

class haversine():
    def rad2deg(self , radians):
        return radians * 180 / pi

    def deg2rad(self , degrees):
        return degrees * pi / 180

    def getDistanceBetweenPointsNew(self , latitude1, longitude1, latitude2, longitude2, unit = 'kilometers'):
        theta = longitude1 - longitude2

        lat1_rad = self.deg2rad(latitude1)
        lat2_rad = self.deg2rad(latitude2)
        theta_rad = self.deg2rad(theta)

        # 原始 cos 值，可能會稍微超出 [-1, 1]
        cos_val = (
            sin(lat1_rad) * sin(lat2_rad) +
            cos(lat1_rad) * cos(lat2_rad) * cos(theta_rad)
        )

        # ✅ 限制 cos_val 落在 [-1, 1]
        cos_val = min(1.0, max(-1.0, cos_val))

        distance = 60 * 1.1515 * self.rad2deg(arccos(cos_val))

        if unit == 'miles':
            return round(distance, 2)
        if unit == 'kilometers':
            return round(distance * 1.609344, 2)
