"""
Analizador de resultados de pruebas de visión artificial
Genera gráficos y métricas de precisión, latencia y errores
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
        
    def load_data(self):
        """Carga el CSV de resultados"""
        try:
            # Cargar CSV
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            print(f"✅ CSV cargado: {len(self.df)} registros")
            print(f"📋 Columnas: {list(self.df.columns)}")
            
            # Convertir latencia y t_total_ms
            if 'latencia' in self.df.columns:
                self.df['latencia_ms'] = self.df['latencia'].apply(self.mmss_to_seconds)
                latencia_validos = self.df['latencia_ms'].notna().sum()
                print(f"   📊 Latencia: {latencia_validos} valores convertidos")
                
            if 't_total_ms' in self.df.columns:
                self.df['t_total_ms_numeric'] = self.df['t_total_ms'].apply(self.mmss_to_seconds)
                t_total_validos = self.df['t_total_ms_numeric'].notna().sum()
                print(f"   📊 t_total_ms: {t_total_validos} valores convertidos")
            
            # Convertir confianza a numérico
            if 'confianza' in self.df.columns:
                self.df['confianza'] = pd.to_numeric(self.df['confianza'], errors='coerce')
                confianza_validos = self.df['confianza'].notna().sum()
                print(f"   📊 Confianza: {confianza_validos} valores convertidos")
            
            # Procesar DEPTH
            print(f"\n📊 Procesando eventos DEPTH...")
            df_depth = self.df[self.df['tipo_evento'] == 'DEPTH'].copy()
            print(f"   Total eventos DEPTH: {len(df_depth)}")
            
            # Extraer distancias DEPTH
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
            
            depth_verdad_count = self.df['distancia_verdad_cm'].notna().sum()
            depth_estimada_count = self.df['distancia_estimada_cm'].notna().sum()
            print(f"   ✅ Distancias verdad: {depth_verdad_count}")
            print(f"   ✅ Distancias estimadas: {depth_estimada_count}")
            
            # Debug: mostrar algunos ejemplos
            if depth_verdad_count > 0:
                ejemplos = self.df[self.df['distancia_verdad_cm'].notna()][
                    ['tipo_evento', 'objeto_verdad', 'distancia_verdad_cm', 
                     'objeto_predicho', 'distancia_estimada_cm']
                ].head(3)
                print("\n   📝 Ejemplos de extracción DEPTH:")
                for idx, row in ejemplos.iterrows():
                    print(f"      Verdad: '{row['objeto_verdad']}' → {row['distancia_verdad_cm']} cm")
                    print(f"      Predicho: '{row['objeto_predicho']}' → {row['distancia_estimada_cm']} cm")
            
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
            
            errores_calculados = self.df['diferencia_distancia_pct'].notna().sum()
            print(f"   ✅ Errores calculados: {errores_calculados}")
            
            # Procesar SCENE
            print(f"\n📊 Procesando eventos SCENE...")
            df_scene = self.df[self.df['tipo_evento'] == 'SCENE'].copy()
            print(f"   Total eventos SCENE: {len(df_scene)}")
            
            # Contar predicciones SCENE con confianza > 0
            scene_con_prediccion = df_scene[
                (df_scene['objeto_predicho'].notna()) & 
                (df_scene['objeto_predicho'] != 'unknown') &
                (df_scene['confianza'].notna()) &
                (df_scene['confianza'] > 0)
            ]
            print(f"   ✅ SCENE con predicción válida: {len(scene_con_prediccion)}")
            
            return True
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {self.csv_path}")
            return False
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
            
            # Buscar número seguido de "cm"
            match = re.search(r'(\d+\.?\d*)\s*cm', dist_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Intentar conversión directa
            try:
                return float(dist_str)
            except:
                return None
                
        except Exception:
            return None
    
    def extract_depth_ground_truth(self, row) -> float:
        """Extrae la distancia verdad de DEPTH"""
        # Primero objeto_verdad
        if pd.notna(row.get('objeto_verdad')):
            obj_verdad = str(row['objeto_verdad']).strip()
            if obj_verdad and obj_verdad.lower() != 'nan':
                dist = self.extract_distance(obj_verdad)
                if dist is not None:
                    return dist
        
        # Si hay columna distancia_m, convertir a cm
        if pd.notna(row.get('distancia_m')):
            try:
                return float(row['distancia_m']) * 100
            except:
                pass
        
        # Intentar del escenario
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
        """Extrae el texto predicho de OCR desde notas o objeto_predicho"""
        # Buscar en notas el resultado de OCR
        notas = row.get('notas')
        if pd.notna(notas):
            notas_str = str(notas)
            # Buscar patrón result: "texto"
            match = re.search(r'result:\s*["\']([^"\']+)["\']', notas_str, re.IGNORECASE | re.DOTALL)
            if match:
                resultado = match.group(1).strip()
                # Limpiar saltos de línea
                resultado = ' '.join(resultado.split())
                return resultado
        
        # Si no, usar objeto_predicho solo si NO es distancia
        if pd.notna(row.get('objeto_predicho')):
            obj_pred = str(row['objeto_predicho']).strip()
            # Verificar que no sea distancia
            if not re.search(r'\d+\.?\d*\s*cm', obj_pred, re.IGNORECASE):
                return obj_pred
        
        return None
    
    def calculate_distance_error(self, real, estimada) -> float:
        """Calcula error porcentual: (|real - estimada| / real) × 100"""
        if pd.isna(real) or pd.isna(estimada) or real == 0:
            return None
        
        try:
            error_pct = (abs(real - estimada) / real) * 100
            return round(error_pct, 2)
        except:
            return None
    
    def classify_distance_error(self, error_pct) -> str:
        """Clasifica error: <20% Acierto, 20-50% Error parcial, >50% Fallo"""
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
    
    def calculate_precision(self):
        """Calcula precisión OCR"""
        print("\n" + "="*60)
        print("📊 PRECISIÓN POR ESCENARIO (OCR)")
        print("="*60)
        
        df_ocr = self.df[self.df['tipo_evento'] == 'OCR'].copy()
        
        if df_ocr.empty:
            print("⚠️ No hay eventos OCR")
            return None
        
        # Extraer ground truth y predicción
        df_ocr['texto_verdad'] = df_ocr.apply(self.extract_ocr_ground_truth, axis=1)
        df_ocr['texto_predicho'] = df_ocr.apply(self.extract_ocr_predicted, axis=1)
        
        df_predictions = df_ocr[
            (df_ocr['texto_verdad'].notna()) & 
            (df_ocr['texto_predicho'].notna())
        ].copy()
        
        if df_predictions.empty:
            print("⚠️ No hay datos de predicción OCR con ground truth")
            print(f"   Total eventos OCR: {len(df_ocr)}")
            print(f"   Con texto_verdad: {df_ocr['texto_verdad'].notna().sum()}")
            print(f"   Con texto_predicho: {df_ocr['texto_predicho'].notna().sum()}")
            
            if df_ocr['texto_verdad'].notna().sum() > 0:
                print("\n   Ejemplos de texto_verdad:")
                for val in df_ocr['texto_verdad'].dropna().head(3):
                    print(f"      '{val}'")
            if df_ocr['texto_predicho'].notna().sum() > 0:
                print("\n   Ejemplos de texto_predicho:")
                for val in df_ocr['texto_predicho'].dropna().head(3):
                    print(f"      '{val}'")
            
            return None
        
        print(f"\n📋 Pares válidos de OCR encontrados: {len(df_predictions)}")
        
        # Mostrar ejemplos
        print("\n📝 Ejemplos de comparación:")
        for idx, row in df_predictions.head(5).iterrows():
            print(f"   Esperado: '{row['texto_verdad']}'")
            print(f"   Obtenido: '{row['texto_predicho']}'")
            print()
        
        df_predictions['acierto_calc'] = df_predictions.apply(
            lambda row: 1 if self.calculate_text_similarity(
                str(row['texto_verdad']), 
                str(row['texto_predicho'])
            ) >= 0.8 else 0, axis=1
        )
        
        precision_por_escenario = df_predictions.groupby('escenario').agg({
            'acierto_calc': ['sum', 'count']
        })
        
        precision_por_escenario.columns = ['verdaderos_positivos', 'total_predicciones']
        precision_por_escenario['falsos_positivos'] = (
            precision_por_escenario['total_predicciones'] - 
            precision_por_escenario['verdaderos_positivos']
        )
        
        precision_por_escenario['precision_pct'] = (
            (precision_por_escenario['verdaderos_positivos'] / 
             (precision_por_escenario['verdaderos_positivos'] + 
              precision_por_escenario['falsos_positivos'])) * 100
        ).round(2)
        
        print(precision_por_escenario)
        
        vp_total = precision_por_escenario['verdaderos_positivos'].sum()
        fp_total = precision_por_escenario['falsos_positivos'].sum()
        precision_general = (vp_total / (vp_total + fp_total) * 100) if (vp_total + fp_total) > 0 else 0
        
        print(f"\n🎯 PRECISIÓN GENERAL: {precision_general:.2f}%")
        print(f"   Verdaderos Positivos: {vp_total}")
        print(f"   Falsos Positivos: {fp_total}")
        
        return precision_por_escenario
    
    def analyze_scene_recognition(self):
        """Analiza precisión de reconocimiento SCENE"""
        print("\n" + "="*60)
        print("🎬 ANÁLISIS DE RECONOCIMIENTO DE ESCENAS (SCENE)")
        print("="*60)
        
        df_scene = self.df[self.df['tipo_evento'] == 'SCENE'].copy()
        
        if df_scene.empty:
            print("⚠️ No hay datos SCENE disponibles")
            return None
        
        # df_scene_valid = df_scene[
        #     (df_scene['objeto_predicho'].notna()) & 
        #     (df_scene['objeto_predicho'] != 'unknown') &
        #     (df_scene['confianza'].notna()) &
        #     (df_scene['confianza'] > 0)
        # ].copy()

        df_with_prediction = df_scene[
            (df_scene['objeto_verdad'].notna()) & 
            (df_scene['objeto_verdad'] != '')
        ].copy()


        df_scene_valid = df_scene[
            (df_scene['objeto_verdad']) == (df_scene['objeto_predicho'])
        ].copy()
        
        total_scene = len(df_with_prediction)
        predicciones_validas = len(df_scene_valid)
        tasa_deteccion = (predicciones_validas / total_scene * 100) if total_scene > 0 else 0
        
        print(f"\n📊 Resumen General:")
        print(f"   Total eventos SCENE: {total_scene}")
        print(f"   Predicciones válidas: {predicciones_validas}")
        print(f"   Tasa de detección: {tasa_deteccion:.2f}%")
        print(f"   Sin detección: {total_scene - predicciones_validas}")
        
        if predicciones_validas > 0:
            confianza_promedio = df_scene_valid['confianza'].mean()
            confianza_min = df_scene_valid['confianza'].min()
            confianza_max = df_scene_valid['confianza'].max()
            
            print(f"\n📈 Estadísticas de Confianza:")
            print(f"   Promedio: {confianza_promedio:.3f}")
            print(f"   Mínima: {confianza_min:.3f}")
            print(f"   Máxima: {confianza_max:.3f}")
            
            print(f"\n🏆 Objetos Detectados:")
            detecciones = df_scene_valid['objeto_predicho'].value_counts()
            for obj, count in detecciones.head(10).items():
                pct = (count / predicciones_validas * 100)
                print(f"   {obj}: {count} ({pct:.1f}%)")
            
            print(f"\n📋 Detección por Escenario:")
            scene_by_scenario = df_scene.groupby('escenario').agg({
                'objeto_predicho': lambda x: (x.notna() & (x != 'unknown')).sum(),
                'id_prueba': 'count'
            })
            scene_by_scenario.columns = ['Detectados', 'Total']
            scene_by_scenario['Tasa_Deteccion_%'] = (
                (scene_by_scenario['Detectados'] / scene_by_scenario['Total']) * 100
            ).round(2)
            print(scene_by_scenario)
        
        return df_scene_valid
    
    def analyze_depth_by_scenario(self):
        """Analiza DEPTH por escenario"""
        print("\n" + "="*60)
        print("📏 ANÁLISIS DETALLADO DEPTH POR ESCENARIO")
        print("="*60)
        
        df_depth = self.df[
            (self.df['tipo_evento'] == 'DEPTH') &
            (self.df['distancia_verdad_cm'].notna()) &
            (self.df['distancia_estimada_cm'].notna())
        ].copy()
        
        if df_depth.empty:
            print("⚠️ No hay datos DEPTH disponibles")
            return None
        
        for escenario in sorted(df_depth['escenario'].unique()):
            df_esc = df_depth[df_depth['escenario'] == escenario]
            
            print(f"\n{'='*60}")
            print(f"🎯 Escenario: {escenario}")
            print(f"{'='*60}")
            
            print(f"Total mediciones: {len(df_esc)}")
            print(f"Distancia real: {df_esc['distancia_verdad_cm'].iloc[0]:.1f} cm")
            print(f"\nMediciones:")
            
            for idx, row in df_esc.iterrows():
                error = row['diferencia_distancia_pct']
                clasificacion = row['clasificacion_distancia']
                emoji = '✅' if clasificacion == 'Acierto' else '⚠️' if clasificacion == 'Error parcial' else '❌'
                
                print(f"  {emoji} Estimada: {row['distancia_estimada_cm']:.1f} cm | "
                      f"Error: {error:.1f}% | {clasificacion}")
            
            print(f"\n📊 Estadísticas:")
            print(f"  Error promedio: {df_esc['diferencia_distancia_pct'].mean():.2f}%")
            print(f"  Error mínimo: {df_esc['diferencia_distancia_pct'].min():.2f}%")
            print(f"  Error máximo: {df_esc['diferencia_distancia_pct'].max():.2f}%")
            print(f"  Desv. estándar: {df_esc['diferencia_distancia_pct'].std():.2f}%")
            
            clasificaciones = df_esc['clasificacion_distancia'].value_counts()
            print(f"\n🏆 Clasificaciones:")
            for clase, count in clasificaciones.items():
                pct = (count / len(df_esc) * 100)
                print(f"  {clase}: {count} ({pct:.1f}%)")
        
        return df_depth
    
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
    
    def analyze_latency_by_mode(self):
        """Analiza latencia por modo"""
        print("\n" + "="*60)
        print("⏱️ LATENCIA POR MODO")
        print("="*60)
        
        df_latency = self.df[self.df['latencia_ms'].notna()].copy()
        
        if df_latency.empty:
            print("⚠️ No hay datos de latencia")
            return None
        
        latency_by_mode = df_latency.groupby('modo')['latencia_ms'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        latency_by_mode.columns = [
            'Media (s)', 'Mediana (s)', 'Desv. Est.', 
            'Mín (s)', 'Máx (s)', 'N° Pruebas'
        ]
        
        print(latency_by_mode)
        return latency_by_mode
    
    def analyze_latency_by_component(self):
        """Analiza latencia por componente"""
        print("\n" + "="*60)
        print("⚙️ LATENCIA POR COMPONENTE")
        print("="*60)
        
        df_component = self.df[self.df['t_total_ms_numeric'].notna()].copy()
        
        if df_component.empty:
            print("⚠️ No hay datos de tiempo por componente")
            return None
        
        latency_by_component = df_component.groupby('tipo_evento')['t_total_ms_numeric'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2)
        
        latency_by_component.columns = [
            'Media (s)', 'Mediana (s)', 'Desv. Est.', 'N° Eventos'
        ]
        latency_by_component = latency_by_component.sort_values('Media (s)', ascending=False)
        
        print(latency_by_component)
        return latency_by_component
    
    def plot_individual_metrics(self):
        """Genera todos los gráficos"""
        print("\n📊 Generando gráficos...")
        
        # 1. Precisión OCR por escenario
        ax1 = plt.subplots(figsize=(18, 12))
        df_ocr = self.df[self.df['tipo_evento'] == 'OCR'].copy()
        if not df_ocr.empty:
            df_ocr['texto_verdad'] = df_ocr.apply(self.extract_ocr_ground_truth, axis=1)
            df_ocr['texto_predicho'] = df_ocr.apply(self.extract_ocr_predicted, axis=1)
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
                ax1.set_title('Precisión por Escenario (OCR)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Escenario')
                ax1.set_ylabel('Precisión (%)')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
                ax1.legend()
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

                plt.savefig(os.path.join(output_dir, 'analisis_precision_por_escenario.png'), dpi=300, bbox_inches='tight')
                print("Gráficos guardados en 'analisis_precision_por_escenario.png'")
            else:
                ax1.text(0.5, 0.5, 'Sin datos OCR válidos', ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes)

                plt.savefig(os.path.join(output_dir, 'analisis_precision_por_escenario.png'), dpi=300, bbox_inches='tight')
                print("Gráficos guardados en 'analisis_precision_por_escenario.png'")
                
        else:
            ax1.text(0.5, 0.5, 'Sin datos OCR', ha='center', va='center', 
                    fontsize=12, transform=ax1.transAxes)

            plt.savefig(os.path.join(output_dir, 'analisis_precision_por_escenario.png'), dpi=300, bbox_inches='tight')
            print("Gráficos guardados en 'analisis_precision_por_escenario.png'")
        
        # 2. Error de distancia DEPTH por escenario
        ax2 = plt.subplots(figsize=(18, 12))
        df_depth = self.df[
            (self.df['tipo_evento'] == 'DEPTH') &
            (self.df['diferencia_distancia_pct'].notna())
        ].copy()
        
        if not df_depth.empty:
            error_means = df_depth.groupby('escenario')['diferencia_distancia_pct'].mean()
            error_means.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
            ax2.set_title('Error Medio de Distancia (DEPTH)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Escenario')
            ax2.set_ylabel('Error (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Umbral Acierto')
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Umbral Fallo')
            ax2.legend()
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.savefig(os.path.join(output_dir, 'error_medio_distancia_depth.png'), dpi=300, bbox_inches='tight')
            print("Gráficos guardados en 'error_medio_distancia_depth.png'")
        else:
            ax2.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
                    
            plt.savefig(os.path.join(output_dir, 'error_medio_distancia_depth.png'), dpi=300, bbox_inches='tight')
            print("Gráficos guardados en 'error_medio_distancia_depth.png'")
        
        # 3. Clasificación de distancias DEPTH
        ax3 = plt.subplot(3, 3, 3)
        if not df_depth.empty:
            classification_counts = df_depth['clasificacion_distancia'].value_counts()
            colors = {'Acierto': 'green', 'Error parcial': 'orange', 'Fallo': 'red'}
            classification_counts.plot(
                kind='pie', ax=ax3, autopct='%1.1f%%',
                colors=[colors.get(x, 'gray') for x in classification_counts.index]
            )
            ax3.set_title('Clasificación de Distancias (DEPTH)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('')
        else:
            ax3.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', fontsize=12)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # 4. Tasa de detección SCENE por escenario
        ax4 = plt.subplot(3, 3, 4)
        df_scene = self.df[self.df['tipo_evento'] == 'SCENE'].copy()
        if not df_scene.empty:
            scene_by_scenario = df_scene.groupby('escenario').apply(
                lambda x: ((x['objeto_predicho'].notna() & 
                           (x['objeto_predicho'] != 'unknown')).sum() / len(x) * 100)
            ).round(2)
            
            scene_by_scenario.plot(kind='bar', ax=ax4, color='mediumpurple', alpha=0.7)
            ax4.set_title('Tasa de Detección SCENE', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Escenario')
            ax4.set_ylabel('Tasa de Detección (%)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Sin datos SCENE', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
        
        # 5. Latencia por componente
        ax5 = plt.subplot(3, 3, 5)
        df_component = self.df[self.df['t_total_ms_numeric'].notna()].copy()
        if not df_component.empty:
            latency_comp = df_component.groupby('tipo_evento')['t_total_ms_numeric'].mean().sort_values(ascending=True)
            latency_comp.plot(kind='barh', ax=ax5, color='teal', alpha=0.7)
            ax5.set_title('Latencia Media por Componente', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Latencia (s)')
            ax5.set_ylabel('Componente')
            ax5.grid(True, alpha=0.3, axis='x')
        else:
            ax5.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax5.transAxes)
        
        # 6. Objetos detectados en SCENE
        ax6 = plt.subplot(3, 3, 6)
        if not df_scene.empty:
            df_scene_valid = df_scene[
                (df_scene['objeto_predicho'].notna()) & 
                (df_scene['objeto_predicho'] != 'unknown')
            ]
            if not df_scene_valid.empty:
                detecciones = df_scene_valid['objeto_predicho'].value_counts().head(10)
                detecciones.plot(kind='bar', ax=ax6, color='indianred', alpha=0.7)
                ax6.set_title('Top 10 Objetos Detectados (SCENE)', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Objeto')
                ax6.set_ylabel('Frecuencia')
                ax6.grid(True, alpha=0.3)
                plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax6.text(0.5, 0.5, 'Sin detecciones', ha='center', va='center', 
                        fontsize=12, transform=ax6.transAxes)
        else:
            ax6.text(0.5, 0.5, 'Sin datos SCENE', ha='center', va='center', 
                    fontsize=12, transform=ax6.transAxes)
        
        # 7. Distribución de latencias
        ax7 = plt.subplot(3, 3, 7)
        df_latency = self.df[self.df['latencia_ms'].notna()]
        if not df_latency.empty:
            for modo in df_latency['modo'].unique():
                data = df_latency[df_latency['modo'] == modo]['latencia_ms']
                ax7.hist(data, bins=20, alpha=0.5, label=modo)
            ax7.set_title('Distribución de Latencias', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Latencia (s)')
            ax7.set_ylabel('Frecuencia')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax7.transAxes)
        
        # 8. Distribución de errores de distancia DEPTH
        ax8 = plt.subplot(3, 3, 8)
        if not df_depth.empty:
            ax8.hist(df_depth['diferencia_distancia_pct'], bins=30, 
                    color='coral', alpha=0.7, edgecolor='black')
            ax8.axvline(x=20, color='g', linestyle='--', linewidth=2, label='Acierto (<20%)')
            ax8.axvline(x=50, color='r', linestyle='--', linewidth=2, label='Fallo (>50%)')
            ax8.set_title('Distribución de Error DEPTH', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Error (%)')
            ax8.set_ylabel('Frecuencia')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', 
                    fontsize=12, transform=ax8.transAxes)
        
        # 9. Distribución de confianza SCENE
        ax9 = plt.subplot(3, 3, 9)
        df_confidence = self.df[
            (self.df['tipo_evento'] == 'SCENE') & 
            (self.df['confianza'].notna()) &
            (self.df['confianza'] > 0)
        ]
        if not df_confidence.empty:
            ax9.hist(df_confidence['confianza'], bins=20, 
                    color='teal', alpha=0.7, edgecolor='black')
            ax9.set_title('Distribución de Confianza (SCENE)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Confianza')
            ax9.set_ylabel('Frecuencia')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax9.transAxes)
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, 'analisis_resultados_completo.png'), dpi=300, bbox_inches='tight')
        # print("✅ Gráficos guardados en 'analisis_resultados_completo.png'")
        # plt.show()

    def plot_all_metrics(self):
        """Genera todos los gráficos"""
        print("\n📊 Generando gráficos...")
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Precisión OCR por escenario
        ax1 = plt.subplot(3, 3, 1)
        df_ocr = self.df[self.df['tipo_evento'] == 'OCR'].copy()
        if not df_ocr.empty:
            df_ocr['texto_verdad'] = df_ocr.apply(self.extract_ocr_ground_truth, axis=1)
            df_ocr['texto_predicho'] = df_ocr.apply(self.extract_ocr_predicted, axis=1)
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
                ax1.set_title('Precisión por Escenario (OCR)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Escenario')
                ax1.set_ylabel('Precisión (%)')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
                ax1.legend()
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax1.text(0.5, 0.5, 'Sin datos OCR válidos', ha='center', va='center', 
                        fontsize=12, transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'Sin datos OCR', ha='center', va='center', 
                    fontsize=12, transform=ax1.transAxes)
        
        # 2. Error de distancia DEPTH por escenario
        ax2 = plt.subplot(3, 3, 2)
        df_depth = self.df[
            (self.df['tipo_evento'] == 'DEPTH') &
            (self.df['diferencia_distancia_pct'].notna())
        ].copy()
        
        if not df_depth.empty:
            error_means = df_depth.groupby('escenario')['diferencia_distancia_pct'].mean()
            error_means.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
            ax2.set_title('Error Medio de Distancia (DEPTH)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Escenario')
            ax2.set_ylabel('Error (%)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Umbral Acierto')
            ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Umbral Fallo')
            ax2.legend()
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
        
        # 3. Clasificación de distancias DEPTH
        ax3 = plt.subplot(3, 3, 3)
        if not df_depth.empty:
            classification_counts = df_depth['clasificacion_distancia'].value_counts()
            colors = {'Acierto': 'green', 'Error parcial': 'orange', 'Fallo': 'red'}
            classification_counts.plot(
                kind='pie', ax=ax3, autopct='%1.1f%%',
                colors=[colors.get(x, 'gray') for x in classification_counts.index]
            )
            ax3.set_title('Clasificación de Distancias (DEPTH)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('')
        else:
            ax3.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', fontsize=12)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
        
        # 4. Tasa de detección SCENE por escenario
        ax4 = plt.subplot(3, 3, 4)
        df_scene = self.df[self.df['tipo_evento'] == 'SCENE'].copy()
        if not df_scene.empty:
            scene_by_scenario = df_scene.groupby('escenario').apply(
                lambda x: ((x['objeto_predicho'].notna() & 
                           (x['objeto_predicho'] != 'unknown')).sum() / len(x) * 100)
            ).round(2)
            
            scene_by_scenario.plot(kind='bar', ax=ax4, color='mediumpurple', alpha=0.7)
            ax4.set_title('Tasa de Detección SCENE', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Escenario')
            ax4.set_ylabel('Tasa de Detección (%)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, 'Sin datos SCENE', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
        
        # 5. Latencia por componente
        ax5 = plt.subplot(3, 3, 5)
        df_component = self.df[self.df['t_total_ms_numeric'].notna()].copy()
        if not df_component.empty:
            latency_comp = df_component.groupby('tipo_evento')['t_total_ms_numeric'].mean().sort_values(ascending=True)
            latency_comp.plot(kind='barh', ax=ax5, color='teal', alpha=0.7)
            ax5.set_title('Latencia Media por Componente', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Latencia (s)')
            ax5.set_ylabel('Componente')
            ax5.grid(True, alpha=0.3, axis='x')
        else:
            ax5.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax5.transAxes)
        
        # 6. Objetos detectados en SCENE
        ax6 = plt.subplot(3, 3, 6)
        if not df_scene.empty:
            df_scene_valid = df_scene[
                (df_scene['objeto_predicho'].notna()) & 
                (df_scene['objeto_predicho'] != 'unknown')
            ]
            if not df_scene_valid.empty:
                detecciones = df_scene_valid['objeto_predicho'].value_counts().head(10)
                detecciones.plot(kind='bar', ax=ax6, color='indianred', alpha=0.7)
                ax6.set_title('Top 10 Objetos Detectados (SCENE)', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Objeto')
                ax6.set_ylabel('Frecuencia')
                ax6.grid(True, alpha=0.3)
                plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax6.text(0.5, 0.5, 'Sin detecciones', ha='center', va='center', 
                        fontsize=12, transform=ax6.transAxes)
        else:
            ax6.text(0.5, 0.5, 'Sin datos SCENE', ha='center', va='center', 
                    fontsize=12, transform=ax6.transAxes)
        
        # 7. Distribución de latencias
        ax7 = plt.subplot(3, 3, 7)
        df_latency = self.df[self.df['latencia_ms'].notna()]
        if not df_latency.empty:
            for modo in df_latency['modo'].unique():
                data = df_latency[df_latency['modo'] == modo]['latencia_ms']
                ax7.hist(data, bins=20, alpha=0.5, label=modo)
            ax7.set_title('Distribución de Latencias', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Latencia (s)')
            ax7.set_ylabel('Frecuencia')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax7.transAxes)
        
        # 8. Distribución de errores de distancia DEPTH
        ax8 = plt.subplot(3, 3, 8)
        if not df_depth.empty:
            ax8.hist(df_depth['diferencia_distancia_pct'], bins=30, 
                    color='coral', alpha=0.7, edgecolor='black')
            ax8.axvline(x=20, color='g', linestyle='--', linewidth=2, label='Acierto (<20%)')
            ax8.axvline(x=50, color='r', linestyle='--', linewidth=2, label='Fallo (>50%)')
            ax8.set_title('Distribución de Error DEPTH', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Error (%)')
            ax8.set_ylabel('Frecuencia')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Sin datos DEPTH', ha='center', va='center', 
                    fontsize=12, transform=ax8.transAxes)
        
        # 9. Distribución de confianza SCENE
        ax9 = plt.subplot(3, 3, 9)
        df_confidence = self.df[
            (self.df['tipo_evento'] == 'SCENE') & 
            (self.df['confianza'].notna()) &
            (self.df['confianza'] > 0)
        ]
        if not df_confidence.empty:
            ax9.hist(df_confidence['confianza'], bins=20, 
                    color='teal', alpha=0.7, edgecolor='black')
            ax9.set_title('Distribución de Confianza (SCENE)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Confianza')
            ax9.set_ylabel('Frecuencia')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Sin datos', ha='center', va='center', 
                    fontsize=12, transform=ax9.transAxes)

        output_dir = "output_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'analisis_resultados_completo.png'), dpi=300, bbox_inches='tight')
        print("✅ Gráficos guardados en 'analisis_resultados_completo.png'")
        plt.show()

    def executive_summary(self):
        """Resumen ejecutivo"""
        print("\n" + "="*70)
        print("📊 RESUMEN EJECUTIVO")
        print("="*70)
        
        # OCR
        df_ocr = self.df[self.df['tipo_evento'] == 'OCR'].copy()
        
        if not df_ocr.empty:
            df_ocr['texto_verdad'] = df_ocr.apply(self.extract_ocr_ground_truth, axis=1)
            df_ocr['texto_predicho'] = df_ocr.apply(self.extract_ocr_predicted, axis=1)
            
            df_ocr_valid = df_ocr[
                (df_ocr['texto_verdad'].notna()) & 
                (df_ocr['texto_predicho'].notna())
            ].copy()
            
            if not df_ocr_valid.empty:
                df_ocr_valid['acierto'] = df_ocr_valid.apply(
                    lambda row: 1 if self.calculate_text_similarity(
                        str(row['texto_verdad']), 
                        str(row['texto_predicho'])
                    ) >= 0.8 else 0, axis=1
                )
                vp_ocr = df_ocr_valid['acierto'].sum()
                total_ocr = len(df_ocr_valid)
                precision_ocr = (vp_ocr / total_ocr * 100) if total_ocr > 0 else 0
                
                print(f"\n🔤 OCR:")
                print(f"   Total pruebas: {total_ocr}")
                print(f"   Aciertos: {vp_ocr}")
                print(f"   Precisión: {precision_ocr:.2f}%")
            else:
                print(f"\n🔤 OCR: {len(df_ocr)} eventos, sin pares válidos")
        else:
            print(f"\n🔤 OCR: Sin datos")
        
        # DEPTH
        df_depth = self.df[
            (self.df['tipo_evento'] == 'DEPTH') &
            (self.df['diferencia_distancia_pct'].notna())
        ]
        
        if not df_depth.empty:
            error_promedio = df_depth['diferencia_distancia_pct'].mean()
            aciertos = (df_depth['clasificacion_distancia'] == 'Acierto').sum()
            total_depth = len(df_depth)
            tasa_acierto = (aciertos / total_depth * 100) if total_depth > 0 else 0
            
            print(f"\n📏 DEPTH:")
            print(f"   Total mediciones: {total_depth}")
            print(f"   Aciertos (<20% error): {aciertos} ({tasa_acierto:.1f}%)")
            print(f"   Error promedio: {error_promedio:.2f}%")
        else:
            print(f"\n📏 DEPTH: Sin datos")
        
        # SCENE
        df_scene = self.df[self.df['tipo_evento'] == 'SCENE']
        if not df_scene.empty:
            scene_valid = df_scene[
                (df_scene['objeto_predicho'].notna()) & 
                (df_scene['objeto_predicho'] != 'unknown') &
                (df_scene['confianza'].notna()) &
                (df_scene['confianza'] > 0)
            ]
            total_scene = len(df_scene)
            detectados = len(scene_valid)
            tasa = (detectados / total_scene * 100) if total_scene > 0 else 0
            
            print(f"\n🎬 SCENE:")
            print(f"   Total eventos: {total_scene}")
            print(f"   Detecciones válidas: {detectados}")
            print(f"   Tasa de detección: {tasa:.2f}%")
        else:
            print(f"\n🎬 SCENE: Sin datos")
        
        # Latencia
        df_latency = self.df[self.df['latencia_ms'].notna()]
        if not df_latency.empty:
            print(f"\n⏱️ LATENCIA:")
            print(f"   Promedio: {df_latency['latencia_ms'].mean():.2f}s")
            print(f"   Mediana: {df_latency['latencia_ms'].median():.2f}s")
        
        print("\n" + "="*70)
    
    def generate_report(self):
        """Genera reporte completo"""
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        print("📋 REPORTE DE ANÁLISIS DE PRUEBAS")
        print("="*60)
        
        self.executive_summary()
        self.calculate_precision()
        self.analyze_depth_by_scenario()
        self.analyze_scene_recognition()
        self.analyze_latency_by_mode()
        self.analyze_latency_by_component()

        # Generar gráficos
        self.plot_all_metrics()
        self.plot_individual_metrics()
        
        print("\n✅ Análisis completado")


if __name__ == "__main__":
    analyzer = TestAnalyzer(csv_path="resultados_pruebas_20260104_161812.csv")
    analyzer.generate_report()