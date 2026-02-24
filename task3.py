import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


DEPTH_PERCENTILE = 1.0
DEPTH_TOLERANCE = 29
MIN_AREA = 370
TILT_THRESHOLD = 0.68
PLY_RADIUS = 0.11
IQR_MULT_XY = 1.6
USE_WEIGHTED_CENTER = True

class PLYReader:
    @staticmethod
    def read_ply(filepath):
        points = []
        try:
            with open(filepath, 'rb') as f:
                header_end = False
                vertex_count = 0
                
                while not header_end:
                    line = f.readline().decode('ascii').strip()
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('end_header'):
                        header_end = True
                
                for _ in range(vertex_count):
                    data = np.frombuffer(f.read(12), dtype=np.float32)
                    if len(data) >= 3:
                        points.append(data[:3])
                    try:
                        f.read(12)
                    except:
                        pass
        except:
            pass
        
        return np.array(points) if points else np.array([])

class ParcelDetector:
    def __init__(self):
        self.fx = 650.0616455078125
        self.fy = 650.0616455078125
        self.cx = 649.5928955078125
        self.cy = 360.9415588378906
        
        self.roi_x = 560
        self.roi_y = 150
        self.roi_w = 300
        self.roi_h = 330
        
        self.ply_reader = PLYReader()
        self.depth_unit = 0.001
        
        self.DEPTH_PERCENTILE = DEPTH_PERCENTILE
        self.DEPTH_TOLERANCE = DEPTH_TOLERANCE
        self.MIN_AREA = MIN_AREA
        self.KERNEL_SIZE = 7
        
        self.Z_STD_THRESHOLD = 0.0035
        self.TILT_THRESHOLD = TILT_THRESHOLD
        self.SNAP_THRESHOLD = 0.995
        
        self.PLY_RADIUS = PLY_RADIUS
        self.PLY_MIN_POINTS = 50
        
        self.IQR_MULT_XY = IQR_MULT_XY
        self.IQR_MULT_Z = 2.0
        
        self.USE_WEIGHTED_CENTER = USE_WEIGHTED_CENTER
        
        print(f"\n{'='*60}")
        print(f"ORIGINAL CODE - BASELINE")
        print(f"{'='*60}")
        print(f"  Use this as baseline for new test set")
        print(f"{'='*60}\n")
    
    def calibrate(self, csv_path, depth_dir):
        try:
            df = pd.read_csv(csv_path)
            samples = []
            
            for _, row in df.iterrows():
                img = row['image_filename']
                x, y, z = row['x'], row['y'], row['z']
                
                d_path = depth_dir / img
                if not d_path.exists():
                    continue
                
                d_img = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
                if d_img is None:
                    continue
                
                u = int(x * self.fx / z + self.cx)
                v = int(y * self.fy / z + self.cy)
                
                if 2 <= u < 1278 and 2 <= v < 718:
                    vals = []
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if 0 <= v+j < 720 and 0 <= u+i < 1280:
                                d = d_img[v+j, u+i]
                                if d > 0:
                                    vals.append(d)
                    
                    if len(vals) >= 5:
                        d_med = np.median(vals)
                        r = d_med / z
                        if 800 < r < 1400:
                            samples.append(r)
            
            if len(samples) >= 10:
                samples = np.array(samples)
                med = np.median(samples)
                mad = np.median(np.abs(samples - med))
                
                if mad > 0:
                    valid = samples[np.abs(samples - med) < 2.5 * mad]
                    if len(valid) >= 8:
                        self.depth_unit = 1.0 / np.median(valid)
                        print(f"Calibrated: depth_unit={self.depth_unit:.6f}")
                        return
            
            print(f"Using default: depth_unit={self.depth_unit}")
        except:
            pass
    
    def to_3d(self, u, v, d):
        if d <= 0:
            return None
        z = d * self.depth_unit
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])
    
    def find_top(self, depth):
        roi = depth[self.roi_y:self.roi_y+self.roi_h,
                   self.roi_x:self.roi_x+self.roi_w].copy()
        
        valid = roi > 0
        if valid.sum() < 100:
            return None
        
        vd = roi[valid]
        mn = np.percentile(vd, self.DEPTH_PERCENTILE)
        
        mask = valid & (roi <= mn + self.DEPTH_TOLERANCE)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.KERNEL_SIZE, self.KERNEL_SIZE))
        m8 = (mask * 255).astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, k)
        
        n, lb, st, _ = cv2.connectedComponentsWithStats(m8, 8)
        if n <= 1:
            return None
        
        valid_comps = []
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area >= self.MIN_AREA:
                valid_comps.append((i, area))
        
        if not valid_comps:
            lg = np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1
        else:
            lg = max(valid_comps, key=lambda x: x[1])[0]
        
        fm = (lb == lg)
        
        ys, xs = np.where(fm)
        if len(xs) < 20:
            return None
        
        pts = []
        for i in range(len(xs)):
            u = self.roi_x + xs[i]
            v = self.roi_y + ys[i]
            pt = self.to_3d(u, v, roi[ys[i], xs[i]])
            if pt is not None and not np.isnan(pt).any():
                pts.append(pt)
        
        if len(pts) < 20:
            return None
        
        pts = np.array(pts)

        for ax in [0, 1]:
            vals = pts[:, ax]
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            if iqr > 0.0005:
                msk = (vals >= q1 - self.IQR_MULT_XY*iqr) & (vals <= q3 + self.IQR_MULT_XY*iqr)
                pts = pts[msk]
                if len(pts) < 20:
                    break
        
        if len(pts) >= 20:
            vals = pts[:, 2]
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            if iqr > 0.0005:
                msk = (vals >= q1 - self.IQR_MULT_Z*iqr) & (vals <= q3 + self.IQR_MULT_Z*iqr)
                pts = pts[msk]
        
        if len(pts) < 20:
            return None

        if self.USE_WEIGHTED_CENTER:
            z_vals = pts[:, 2]
            z_min = np.min(z_vals)
            weights = z_vals - z_min + 0.001
            weights = weights / np.sum(weights)
            c = np.average(pts, axis=0, weights=weights)
        else:
            c = np.mean(pts, axis=0)
        
        return {'centroid': c, 'points': pts}
    
    def normal(self, pts):
        if len(pts) < 20:
            return np.array([0.0, 0.0, 1.0])
        
        zs = np.std(pts[:, 2])
        if zs < self.Z_STD_THRESHOLD:
            return np.array([0.0, 0.0, 1.0])
        
        try:
            X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
            A = np.column_stack([X, Y, np.ones(len(X))])
            p, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            
            n = np.array([-p[0], -p[1], 1.0])
            n = n / np.linalg.norm(n)
            
            if n[2] < 0:
                n = -n
            if n[2] < self.TILT_THRESHOLD:
                return np.array([0.0, 0.0, 1.0])
            if n[2] > self.SNAP_THRESHOLD:
                return np.array([0.0, 0.0, 1.0])
            
            return n
        except:
            return np.array([0.0, 0.0, 1.0])
    
    def normal_ply(self, ply_path, c):
        try:
            pts = self.ply_reader.read_ply(ply_path)
            if len(pts) < 50:
                return None
            
            v = ~(np.isnan(pts).any(axis=1) | np.isinf(pts).any(axis=1))
            pts = pts[v]
            if len(pts) < 50:
                return None
            
            ds = np.linalg.norm(pts - c, axis=1)
            
            for radius in [self.PLY_RADIUS, self.PLY_RADIUS * 0.85, self.PLY_RADIUS * 1.15]:
                nb = pts[ds < radius]
                
                if len(nb) >= self.PLY_MIN_POINTS:
                    zm = np.median(nb[:, 2])
                    zs = np.std(nb[:, 2])
                    if zs > 0.001:
                        nb = nb[np.abs(nb[:, 2] - zm) < 2.5*zs]
                    
                    if len(nb) >= 40:
                        for ax in [0, 1]:
                            vals = nb[:, ax]
                            med, std = np.median(vals), np.std(vals)
                            if std > 0.001:
                                nb = nb[np.abs(vals - med) < 2.5*std]
                    
                    if len(nb) >= 35:
                        return self.normal(nb)
            
            return None
        except:
            return None
    
    def process(self, d_path, p_path=None):
        d = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        
        r = self.find_top(d)
        if not r:
            return None
        
        c = r['centroid']
        pts = r['points']
        
        n = None
        if p_path and Path(p_path).exists():
            n = self.normal_ply(p_path, c)
        
        if n is None:
            n = self.normal(pts)
        
        if n is None or np.isnan(n).any():
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / np.linalg.norm(n)
        
        return {
            'x': float(c[0]), 'y': float(c[1]), 'z': float(c[2]),
            'Rx': float(n[0]), 'Ry': float(n[1]), 'Rz': float(n[2])
        }

def main():
    det = ParcelDetector()
    b = Path('.')
    
    td = None
    for p in [b / 'dataset' / 'test0', b / 'test0']:
        if p.exists() and (p / 'rgb').exists():
            td = p
            break
    
    if not td:
        pd.DataFrame(columns=['image_filename','x','y','z','Rx','Ry','Rz']).to_csv('Submission_3D.csv', index=False)
        return
    
    for tr in [b / 'dataset' / 'train', b / 'train']:
        if tr.exists():
            csv = tr / 'Public train.csv'
            if csv.exists():
                det.calibrate(csv, tr / 'depth')
                break
    
    fs = sorted(list((td / 'rgb').glob('*.png')), key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or '0'))
    
    print(f"Processing {len(fs)} test images...\n")
    
    res = []
    for i, f in enumerate(fs):
        nm = f.name
        out = f"image_{f.stem}{f.suffix}" if nm[0].isdigit() else nm
        
        dp = td / 'depth' / nm
        if not dp.exists():
            dp = td / 'depth' / (f.stem + '.png')
        
        pp = td / 'ply' / (f.stem + '.ply')
        if not pp.exists():
            pp = None
        
        r = det.process(dp, pp)
        
        if r:
            normal = np.array([r['Rx'], r['Ry'], r['Rz']])
            norm_check = np.linalg.norm(normal)
            if abs(norm_check - 1.0) > 0.001:
                normal = normal / norm_check
                r['Rx'], r['Ry'], r['Rz'] = normal[0], normal[1], normal[2]
            
            res.append({
                'image_filename': out,
                'x': round(r['x'], 3), 'y': round(r['y'], 3), 'z': round(r['z'], 3),
                'Rx': round(r['Rx'], 4), 'Ry': round(r['Ry'], 4), 'Rz': round(r['Rz'], 4)
            })
        else:
            res.append({
                'image_filename': out,
                'x': 0.0, 'y': 0.0, 'z': 1.0,
                'Rx': 0.0, 'Ry': 0.0, 'Rz': 1.0
            })
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(fs)}")
    
    pd.DataFrame(res).to_csv('Submission_3D.csv', index=False, encoding='utf-8', lineterminator='\n')
    
    print(f"\nDone!\n")

if __name__ == "__main__":
    main()