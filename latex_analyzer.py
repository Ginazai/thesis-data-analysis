"""
Analizador de resultados de pruebas de visión artificial
Genera gráficos, métricas y tablas LaTeX para el reporte
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import os

output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

class TestAnalyzer:
    def __init__(self, csv_path: str = "resultados_pruebas_20260104_161812.csv"):
        self.csv_path = csv_path
        self.df = None
        self.metrics = {}
        
    def load_data(self):
        """Carga el CSV de resultados"""
        try:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            print(f"✅ CSV cargado: {len(self.df)} registros")
            print(f"📋 Columnas: {list(self.df.columns)}")
            
            # Convertir latencia y t_total_ms
            if 'latencia' in self.df.columns:
                self.df['latencia_ms'] = self.df['latencia'].apply(self.mmss_to_seconds)
                
            if 't_total_ms' in self.df.columns:
                self.df['t_total_ms_numeric'] = self.df['t_total_ms'].apply(self.mmss_to_seconds)
            
            # Convertir confianza a numérico
            if 'confianza' in self.df.columns:
                self.df['confianza'] = pd.to_numeric(self.df['confianza'], errors='coerce')
            
            # Procesar DEPTH
            self.df['distancia_verdad_cm'] = self.df.apply(
                lambda row: self.extract_depth_ground_truth(row) 
                if row.get('tipo_evento') == 'DEPTH' else None, 
                axis=1
            )
            
            self.df['distancia_estimada_cm'] = self.df.apply(
                lambda row: self.extract_distance(row.get('objeto_predicho')) 
                if row.get('tipo_evento') == 'DEPTH' else None, 
                axis=1
            )
            
            # Calcular diferencias de distancia
            self.df['diferencia_distancia_pct'] = self.df.apply(
                lambda row: self.calculate_distance_error(
                    row.get('distancia_verdad_cm'), 
                    row.get('distancia_estimada_cm')
                ), axis=1
            )
            
            self.df['clasificacion_distancia'] = self.df['diferencia_distancia_pct'].apply(
                self.classify_distance_error
            )
            
            # Procesar OCR
            self.df['texto_verdad'] = self.df.apply(
                lambda row: self.extract_ocr_ground_truth(row) 
                if row.get('tipo_evento') == 'OCR' else None, 
                axis=1
            )
            
            self.df['texto_predicho'] = self.df.apply(
                lambda row: self.extract_ocr_predicted(row) 
                if row.get('tipo_evento') == 'OCR' else None, 
                axis=1
            )
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_distance(self, dist_str) -> float:
        """Extrae valor numérico de distancia en cm"""
        if pd.isna(dist_str):
            return None
        
        try:
            dist_str = str(dist_str).strip()
            if not dist_str or dist_str in ['', 'nan', 'None']:
                return None
            
            match = re.search(r'(\d+\.?\d*)\s*cm', dist_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            try:
                return float(dist_str)
            except:
                return None
        except Exception:
            return None
    
    def extract_depth_ground_truth(self, row) -> float:
        """Extrae la distancia verdad de DEPTH"""
        if pd.notna(row.get('objeto_verdad')):
            obj_verdad = str(row['objeto_verdad']).strip()
            if obj_verdad and obj_verdad.lower() != 'nan':
                dist = self.extract_distance(obj_verdad)
                if dist is not None:
                    return dist
        
        if pd.notna(row.get('distancia_m')):
            try:
                return float(row['distancia_m']) * 100
            except:
                pass
        
        escenario = row.get('escenario', '')
        if pd.notna(escenario):
            escenario_str = str(escenario)
            match = re.search(r'(\d+\.?\d*)\s*cm', escenario_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            match = re.search(r'(\d+\.?\d*)\s*m\b', escenario_str, re.IGNORECASE)
            if match:
                return float(match.group(1)) * 100
        
        return None
    
    def extract_ocr_ground_truth(self, row) -> str:
        """Extrae el texto verdad de OCR"""
        if pd.notna(row.get('objeto_verdad')):
            obj_verdad = str(row['objeto_verdad']).strip()
            if obj_verdad and obj_verdad.lower() not in ['nan', '', 'none']:
                return obj_verdad
        return None
    
    def extract_ocr_predicted(self, row) -> str:
        """Extrae el texto predicho de OCR"""
        notas = row.get('notas')
        if pd.notna(notas):
            notas_str = str(notas)
            match = re.search(r'result:\s*["\']([^"\']+)["\']', notas_str, re.IGNORECASE | re.DOTALL)
            if match:
                resultado = match.group(1).strip()
                resultado = ' '.join(resultado.split())
                return resultado
        
        if pd.notna(row.get('objeto_predicho')):
            obj_pred = str(row['objeto_predicho']).strip()
            if not re.search(r'\d+\.?\d*\s*cm', obj_pred, re.IGNORECASE):
                return obj_pred
        
        return None
    
    def calculate_distance_error(self, real, estimada) -> float:
        """Calcula error porcentual"""
        if pd.isna(real) or pd.isna(estimada) or real == 0:
            return None
        
        try:
            error_pct = (abs(real - estimada) / real) * 100
            return round(error_pct, 2)
        except:
            return None
    
    def classify_distance_error(self, error_pct) -> str:
        """Clasifica error de distancia"""
        if pd.isna(error_pct):
            return None
        
        if error_pct < 20:
            return 'Acierto'
        elif error_pct <= 50:
            return 'Error parcial'
        else:
            return 'Fallo'
    
    def mmss_to_seconds(self, time_str: str) -> float:
        """Convierte MM:SS.s o HH:MM:SS.s a segundos"""
        if pd.isna(time_str):
            return None

        try:
            time_str = str(time_str).strip()
            if time_str == "":
                return None

            is_negative = time_str.startswith("-")
            if is_negative:
                time_str = time_str[1:]

            parts = time_str.split(":")

            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                total_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                return None

            return -total_seconds if is_negative else total_seconds

        except Exception:
            return None
    
    def calculate_text_similarity(self, ground_truth: str, predicted: str) -> float:
        """Calcula similitud usando distancia de Levenshtein"""
        gt = str(ground_truth).lower().strip()
        pred = str(predicted).lower().strip()
        
        if gt == pred:
            return 1.0
        
        if not gt or not pred:
            return 0.0
        
        distance = self.levenshtein_distance(gt, pred)
        max_len = max(len(gt), len(pred))
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Distancia de Levenshtein"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_all_metrics(self):
        """Calcula todas las métricas necesarias para las tablas"""
        print("\n" + "="*60)
        print("📊 CALCULANDO MÉTRICAS PARA TABLAS")
        print("="*60)
        
        metrics = {
            'LOCAL': {},
            'MOBILE_SCENE': {},
            'MOBILE_OCR': {},
            'MOBILE_DEPTH': {},
            'MOBILE_GENERAL': {}
        }
        
        # === MODO LOCAL ===
        df_local = self.df[self.df['modo'] == 'LOCAL']
        if not df_local.empty:
            metrics['LOCAL'] = self._calculate_mode_metrics(df_local, 'LOCAL')
        else:
            print("⚠️ No hay datos de Modo Local")
        
        # === MODO HÍBRIDO - SCENE ===
        df_scene = self.df[(self.df['modo'] == 'MOBILE') & (self.df['tipo_evento'] == 'SCENE')]
        if not df_scene.empty:
            metrics['MOBILE_SCENE'] = self._calculate_scene_metrics(df_scene)
        
        # === MODO HÍBRIDO - OCR ===
        df_ocr = self.df[(self.df['modo'] == 'MOBILE') & (self.df['tipo_evento'] == 'OCR')]
        if not df_ocr.empty:
            metrics['MOBILE_OCR'] = self._calculate_ocr_metrics(df_ocr)
        
        # === MODO HÍBRIDO - DEPTH ===
        df_depth = self.df[(self.df['modo'] == 'MOBILE') & (self.df['tipo_evento'] == 'DEPTH')]
        if not df_depth.empty:
            metrics['MOBILE_DEPTH'] = self._calculate_depth_metrics(df_depth)
        
        # === MODO HÍBRIDO GENERAL ===
        df_mobile = self.df[self.df['modo'] == 'MOBILE']
        if not df_mobile.empty:
            metrics['MOBILE_GENERAL'] = self._calculate_mode_metrics(df_mobile, 'MOBILE')
        
        self.metrics = metrics
        return metrics
    
    def _calculate_mode_metrics(self, df, mode_name):
        """Calcula métricas generales para un modo"""
        metrics = {
            'n_pruebas': len(df),
            'aciertos': 0,
            'errores': 0,
            'precision_pct': 0.0,
            'fallos': 0,
            'tasa_fallos_pct': 0.0
        }
        
        # Latencia
        df_lat = df[df['latencia_ms'].notna()]
        if not df_lat.empty:
            metrics['latencia_media_ms'] = df_lat['latencia_ms'].mean() * 1000
            metrics['latencia_sd_ms'] = df_lat['latencia_ms'].std() * 1000
            metrics['latencia_min_ms'] = df_lat['latencia_ms'].min() * 1000
            metrics['latencia_max_ms'] = df_lat['latencia_ms'].max() * 1000
        
        return metrics
    
    def _calculate_scene_metrics(self, df_scene):
        """Calcula métricas específicas de SCENE"""
        # Predicciones válidas: cuando hay objeto_verdad Y objeto_predicho coinciden
        df_valid = df_scene[
            (df_scene['objeto_verdad'].notna()) & 
            (df_scene['objeto_verdad'] != '')
        ].copy()
        
        total = len(df_valid)
        aciertos = (df_valid['objeto_verdad'] == df_valid['objeto_predicho']).sum()
        errores = total - aciertos
        precision = (aciertos / total * 100) if total > 0 else 0
        
        # Latencia
        df_lat = df_scene[df_scene['t_total_ms_numeric'].notna()]
        
        metrics = {
            'n_pruebas': total,
            'aciertos': int(aciertos),
            'errores': int(errores),
            'precision_pct': round(precision, 2),
            'fallos': 0,  # Se calculará después
            'tasa_fallos_pct': 0.0
        }
        
        if not df_lat.empty:
            metrics['latencia_media_ms'] = df_lat['t_total_ms_numeric'].mean() * 1000
            metrics['latencia_sd_ms'] = df_lat['t_total_ms_numeric'].std() * 1000
            metrics['latencia_min_ms'] = df_lat['t_total_ms_numeric'].min() * 1000
            metrics['latencia_max_ms'] = df_lat['t_total_ms_numeric'].max() * 1000
        
        return metrics
    
    def _calculate_ocr_metrics(self, df_ocr):
        """Calcula métricas específicas de OCR"""
        df_valid = df_ocr[
            (df_ocr['texto_verdad'].notna()) & 
            (df_ocr['texto_predicho'].notna())
        ].copy()
        
        if df_valid.empty:
            return {'n_pruebas': 0, 'aciertos': 0, 'errores': 0, 'precision_pct': 0.0}
        
        df_valid['acierto'] = df_valid.apply(
            lambda row: 1 if self.calculate_text_similarity(
                str(row['texto_verdad']), str(row['texto_predicho'])
            ) >= 0.8 else 0, axis=1
        )
        
        total = len(df_valid)
        aciertos = df_valid['acierto'].sum()
        errores = total - aciertos
        precision = (aciertos / total * 100) if total > 0 else 0
        
        # Latencia
        df_lat = df_ocr[df_ocr['t_total_ms_numeric'].notna()]
        
        metrics = {
            'n_pruebas': total,
            'aciertos': int(aciertos),
            'errores': int(errores),
            'precision_pct': round(precision, 2),
            'fallos': 0,
            'tasa_fallos_pct': 0.0
        }
        
        if not df_lat.empty:
            metrics['latencia_media_ms'] = df_lat['t_total_ms_numeric'].mean() * 1000
            metrics['latencia_sd_ms'] = df_lat['t_total_ms_numeric'].std() * 1000
            metrics['latencia_min_ms'] = df_lat['t_total_ms_numeric'].min() * 1000
            metrics['latencia_max_ms'] = df_lat['t_total_ms_numeric'].max() * 1000
        
        return metrics
    
    def _calculate_depth_metrics(self, df_depth):
        """Calcula métricas específicas de DEPTH"""
        df_valid = df_depth[
            (df_depth['distancia_verdad_cm'].notna()) & 
            (df_depth['distancia_estimada_cm'].notna())
        ].copy()
        
        if df_valid.empty:
            return {'n_pruebas': 0, 'aciertos': 0, 'errores': 0, 'precision_pct': 0.0}
        
        total = len(df_valid)
        aciertos = (df_valid['clasificacion_distancia'] == 'Acierto').sum()
        errores = total - aciertos
        precision = (aciertos / total * 100) if total > 0 else 0
        
        # Latencia
        df_lat = df_depth[df_depth['t_total_ms_numeric'].notna()]
        
        metrics = {
            'n_pruebas': total,
            'aciertos': int(aciertos),
            'errores': int(errores),
            'precision_pct': round(precision, 2),
            'fallos': 0,
            'tasa_fallos_pct': 0.0
        }
        
        if not df_lat.empty:
            metrics['latencia_media_ms'] = df_lat['t_total_ms_numeric'].mean() * 1000
            metrics['latencia_sd_ms'] = df_lat['t_total_ms_numeric'].std() * 1000
            metrics['latencia_min_ms'] = df_lat['t_total_ms_numeric'].min() * 1000
            metrics['latencia_max_ms'] = df_lat['t_total_ms_numeric'].max() * 1000
        
        return metrics
    
    def generate_latex_tables(self):
        """Genera todas las tablas LaTeX"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("📄 GENERANDO TABLAS LATEX")
        print("="*60)
        
        tables = []
        
        # Tabla 1: Precisión global
        tables.append(self._generate_precision_global_table())
        
        # Tabla 2: Precisión por escenario
        tables.append(self._generate_precision_escenario_table())
        
        # Tabla 3: Latencia del sistema
        tables.append(self._generate_latencia_sistema_table())
        
        # Tabla 4: Latencia por componentes
        tables.append(self._generate_latencia_componentes_table())
        
        # Tabla 5: Tasa de fallos
        tables.append(self._generate_tasa_fallos_table())
        
        # Guardar en archivo
        output_file = os.path.join(output_dir, 'tablas_latex.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(tables))
        
        print(f"\n✅ Tablas guardadas en: {output_file}")
        
        # Mostrar en consola
        for table in tables:
            print("\n" + table)
        
        return tables
    
    def _generate_precision_global_table(self):
        """Tabla 1: Precisión global"""
        m = self.metrics
        
        def get_val(key, field, default='N/A'):
            return m.get(key, {}).get(field, default)
        
        table = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Modalidad} & \textbf{N° pruebas} & \textbf{Aciertos} & \textbf{Errores} & \textbf{Precisión (\%)} \\
\midrule"""
        
        table += f"\nModo Local & {get_val('LOCAL', 'n_pruebas', 0)} & {get_val('LOCAL', 'aciertos', 0)} & {get_val('LOCAL', 'errores', 0)} & {get_val('LOCAL', 'precision_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - Scene & {get_val('MOBILE_SCENE', 'n_pruebas', 0)} & {get_val('MOBILE_SCENE', 'aciertos', 0)} & {get_val('MOBILE_SCENE', 'errores', 0)} & {get_val('MOBILE_SCENE', 'precision_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - OCR & {get_val('MOBILE_OCR', 'n_pruebas', 0)} & {get_val('MOBILE_OCR', 'aciertos', 0)} & {get_val('MOBILE_OCR', 'errores', 0)} & {get_val('MOBILE_OCR', 'precision_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - Depth & {get_val('MOBILE_DEPTH', 'n_pruebas', 0)} & {get_val('MOBILE_DEPTH', 'aciertos', 0)} & {get_val('MOBILE_DEPTH', 'errores', 0)} & {get_val('MOBILE_DEPTH', 'precision_pct', 0):.2f} \\\\"
        
        table += r"""
\bottomrule
\end{tabular}
\caption{Precisión global del prototipo por modalidad}
\label{tab:precision-global}
\end{table}"""
        
        return table
    
    def _generate_precision_escenario_table(self):
        """Tabla 2: Precisión por escenario"""
        # Calcular por escenario
        escenarios = {}
        
        for modo in ['LOCAL', 'MOBILE']:
            df_modo = self.df[self.df['modo'] == modo]
            if df_modo.empty:
                continue
            
            for escenario in df_modo['escenario'].unique():
                if pd.isna(escenario):
                    continue
                    
                df_esc = df_modo[df_modo['escenario'] == escenario]
                
                # Para SCENE
                df_scene = df_esc[df_esc['tipo_evento'] == 'SCENE']
                if not df_scene.empty:
                    df_valid = df_scene[
                        (df_scene['objeto_verdad'].notna()) & 
                        (df_scene['objeto_verdad'] != '')
                    ]
                    if not df_valid.empty:
                        aciertos = (df_valid['objeto_verdad'] == df_valid['objeto_predicho']).sum()
                        precision = (aciertos / len(df_valid) * 100) if len(df_valid) > 0 else 0
                        escenarios[f"{escenario}_{modo}_SCENE"] = round(precision, 2)
        
        table = r"""\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{Xcc}
\toprule
\textbf{Escenario de prueba} & \textbf{Modo Local (\%)} & \textbf{Modo Híbrido - Scene (\%)} \\
\midrule"""
        
        # Agrupar por escenario base
        esc_unicos = sorted(set([k.split('_')[0] for k in escenarios.keys()]))
        for esc in esc_unicos:
            local_val = escenarios.get(f"{esc}_LOCAL_SCENE", 'N/A')
            mobile_val = escenarios.get(f"{esc}_MOBILE_SCENE", 'N/A')
            table += f"\nEscenario {esc} & {local_val} & {mobile_val} \\\\"
        
        table += r"""
\bottomrule
\end{tabularx}
\caption{Precisión por escenario de prueba}
\label{tab:precision-escenario}
\end{table}"""
        
        return table
    
    def _generate_latencia_sistema_table(self):
        """Tabla 3: Latencia del sistema"""
        m = self.metrics
        
        def get_lat(key, field, default=0):
            val = m.get(key, {}).get(field, default)
            return f"{val:.0f}" if val != 0 else "N/A"
        
        table = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Modalidad} & \textbf{Media (ms)} & \textbf{SD (ms)} & \textbf{Mín (ms)} & \textbf{Máx (ms)} \\
\midrule"""
        
        table += f"\nModo Local & {get_lat('LOCAL', 'latencia_media_ms')} & {get_lat('LOCAL', 'latencia_sd_ms')} & {get_lat('LOCAL', 'latencia_min_ms')} & {get_lat('LOCAL', 'latencia_max_ms')} \\\\"
        table += f"\nModo Híbrido - Scene & {get_lat('MOBILE_SCENE', 'latencia_media_ms')} & {get_lat('MOBILE_SCENE', 'latencia_sd_ms')} & {get_lat('MOBILE_SCENE', 'latencia_min_ms')} & {get_lat('MOBILE_SCENE', 'latencia_max_ms')} \\\\"
        table += f"\nModo Híbrido - OCR & {get_lat('MOBILE_OCR', 'latencia_media_ms')} & {get_lat('MOBILE_OCR', 'latencia_sd_ms')} & {get_lat('MOBILE_OCR', 'latencia_min_ms')} & {get_lat('MOBILE_OCR', 'latencia_max_ms')} \\\\"
        table += f"\nModo Híbrido - Depth & {get_lat('MOBILE_DEPTH', 'latencia_media_ms')} & {get_lat('MOBILE_DEPTH', 'latencia_sd_ms')} & {get_lat('MOBILE_DEPTH', 'latencia_min_ms')} & {get_lat('MOBILE_DEPTH', 'latencia_max_ms')} \\\\"
        
        table += r"""
\bottomrule
\end{tabular}
\caption{Latencia del sistema por modalidad}
\label{tab:latencia-sistema}
\end{table}"""
        
        return table
    
    def _generate_latencia_componentes_table(self):
        """Tabla 4: Latencia por componentes"""
        # Calcular tiempos por componente
        componentes = ['ESP32', 'SCENE', 'OCR', 'DEPTH', 'TTS']
        tiempos = {}
        
        for modo in ['LOCAL', 'MOBILE']:
            tiempos[modo] = {}
            for comp in componentes:
                df_comp = self.df[
                    (self.df['modo'] == modo) & 
                    (self.df['tipo_evento'] == comp) &
                    (self.df['t_total_ms_numeric'].notna())
                ]
                if not df_comp.empty:
                    tiempos[modo][comp] = df_comp['t_total_ms_numeric'].mean() * 1000
                else:
                    tiempos[modo][comp] = 0
        
        table = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Componente} & \textbf{Modo Local (ms)} & \textbf{Scene (ms)} & \textbf{OCR (ms)} & \textbf{Depth (ms)} \\
\midrule"""
        
        comp_names = {
            'ESP32': 'Captura de imagen',
            'SCENE': 'Inferencia Scene',
            'OCR': 'Inferencia OCR',
            'DEPTH': 'Inferencia Depth',
            'TTS': 'Generación audio'
        }
        
        for comp in componentes:
            local_val = f"{tiempos['LOCAL'].get(comp, 0):.0f}" if tiempos['LOCAL'].get(comp, 0) > 0 else "N/A"
            scene_val = f"{tiempos['MOBILE'].get(comp, 0):.0f}" if tiempos['MOBILE'].get(comp, 0) > 0 else "N/A"
            table += f"\n{comp_names.get(comp, comp)} & {local_val} & {scene_val} & {scene_val} & {scene_val} \\\\"
        
        table += r"""
\bottomrule
\end{tabular}
\caption{Desglose de latencia por componente del sistema}
\label{tab:latencia-componentes}
\end{table}"""
        
        return table
    
    def _generate_tasa_fallos_table(self):
        """Tabla 5: Tasa de fallos"""
        m = self.metrics
        
        def get_val(key, field, default=0):
            return m.get(key, {}).get(field, default)
        
        table = r"""\begin{table}[H]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Modalidad} & \textbf{Pruebas totales} & \textbf{Fallos} & \textbf{Tasa de fallos (\%)} \\
\midrule"""
        
        table += f"\nModo Local & {get_val('LOCAL', 'n_pruebas', 0)} & {get_val('LOCAL', 'fallos', 0)} & {get_val('LOCAL', 'tasa_fallos_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - Scene & {get_val('MOBILE_SCENE', 'n_pruebas', 0)} & {get_val('MOBILE_SCENE', 'fallos', 0)} & {get_val('MOBILE_SCENE', 'tasa_fallos_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - OCR & {get_val('MOBILE_OCR', 'n_pruebas', 0)} & {get_val('MOBILE_OCR', 'fallos', 0)} & {get_val('MOBILE_OCR', 'tasa_fallos_pct', 0):.2f} \\\\"
        table += f"\nModo Híbrido - Depth & {get_val('MOBILE_DEPTH', 'n_pruebas', 0)} & {get_val('MOBILE_DEPTH', 'fallos', 0)} & {get_val('MOBILE_DEPTH', 'tasa_fallos_pct', 0):.2f} \\\\"
        
        table += r"""
\bottomrule
\end{tabular}
\caption{Tasa de fallos por modalidad}
\label{tab:robustez}
\end{table}"""
        
        return table
    
    def plot_individual_metrics(self):
        """Genera gráficos individuales"""
        print("\n📊 Generando gráficos individuales...")
        
        # 1. Precisión OCR
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        df_ocr = self.df[self.df['tipo_evento'] == 'OCR'].copy()
        df_ocr_valid = df_ocr[
            (df_ocr['texto_verdad'].notna()) & 
            (df_ocr['texto_predicho'].notna())
        ].copy()
        
        if not df_ocr_valid.empty:
            df_ocr_valid['acierto'] = df_ocr_valid.apply(
                lambda row: 1 if self.calculate_text_similarity(
                    str(row['texto_verdad']), str(row['texto_predicho'])
                ) >= 0.8 else 0, axis=1
            )
            precision_data = df_ocr_valid.groupby('escenario')['acierto'].apply(
                lambda x: (x.sum() / len(x) * 100)
            ).round(2)
            
            precision_data.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
            ax1.set_title('Precisión por Escenario (OCR)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Escenario')
            ax1.set_ylabel('Precisión (%)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
            ax1.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '01_precision_ocr.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Error DEPTH
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        df_depth = self.df[
            (self.df['tipo_evento'] == 'DEPTH') &
            (self.df['diferencia_distancia_pct'].notna())
        ].copy()
        
        if not df_depth.empty:
            error_means = df_depth.groupby('escenario')['diferencia_distancia_pct'].mean()
            error_means.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
            ax2.set_title('Error Medio de Distancia (DEPTH)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Escenario')
            ax2.set_ylabel('Error (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Umbral Acierto (20%)')
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Umbral Fallo (50%)')
            ax2.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '02_error_depth.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Clasificación DEPTH (pie chart)
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        if not df_depth.empty:
            classification_counts = df_depth['clasificacion_distancia'].value_counts()
            colors = {'Acierto': 'green', 'Error parcial': 'orange', 'Fallo': 'red'}
            ax3.pie(classification_counts, labels=classification_counts.index,
                   autopct='%1.1f%%', colors=[colors.get(x, 'gray') for x in classification_counts.index],
                   startangle=90)
            ax3.set_title('Clasificación de Distancias (DEPTH)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '03_clasificacion_depth.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Tasa detección SCENE
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        df_scene = self.df[self.df['tipo_evento'] == 'SCENE'].copy()
        if not df_scene.empty:
            scene_by_scenario = df_scene.groupby('escenario').apply(
                lambda x: ((x['objeto_predicho'].notna() & 
                           (x['objeto_predicho'] != 'unknown')).sum() / len(x) * 100)
            ).round(2)
            
            scene_by_scenario.plot(kind='bar', ax=ax4, color='mediumpurple', alpha=0.7)
            ax4.set_title('Tasa de Detección SCENE por Escenario', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Escenario')
            ax4.set_ylabel('Tasa de Detección (%)')
            ax4.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '04_tasa_deteccion_scene.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Latencia por componente
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        df_component = self.df[self.df['t_total_ms_numeric'].notna()].copy()
        if not df_component.empty:
            latency_comp = df_component.groupby('tipo_evento')['t_total_ms_numeric'].mean().sort_values(ascending=True)
            latency_comp.plot(kind='barh', ax=ax5, color='teal', alpha=0.7)
            ax5.set_title('Latencia Media por Componente', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Latencia (s)')
            ax5.set_ylabel('Componente')
            ax5.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '05_latencia_componente.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Distribución latencias
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        df_latency = self.df[self.df['latencia_ms'].notna()]
        if not df_latency.empty:
            for modo in df_latency['modo'].unique():
                data = df_latency[df_latency['modo'] == modo]['latencia_ms']
                ax6.hist(data, bins=20, alpha=0.5, label=modo)
            ax6.set_title('Distribución de Latencias', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Latencia (s)')
            ax6.set_ylabel('Frecuencia')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '06_distribucion_latencias.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Gráficos individuales guardados en '{output_dir}'")
    
    def generate_report(self):
        """Genera reporte completo"""
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        print("📋 REPORTE DE ANÁLISIS COMPLETO")
        print("="*60)
        
        # Calcular métricas
        self.calculate_all_metrics()
        
        # Generar tablas LaTeX
        self.generate_latex_tables()
        
        # Generar gráficos
        self.plot_individual_metrics()
        
        print("\n✅ Análisis completado")


if __name__ == "__main__":
    analyzer = TestAnalyzer(csv_path="resultados_pruebas_20260104_161812.csv")
    analyzer.generate_report()
