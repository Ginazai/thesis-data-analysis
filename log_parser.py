"""
Parser de logs de pruebas de visión artificial a CSV
Procesa archivos .log/.txt y genera un CSV estructurado con métricas de pruebas
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

class LogParser:
    def __init__(self, logs_directory: str = "./logs"):
        self.logs_dir = Path(logs_directory)
        self.id_counter = 1
        
    def parse_mobile_line(self, line: str) -> Optional[Dict]:
        """Parsea línea formato MOBILE: [ISO8601] COMP: event - details"""
        pattern = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]\s+(\w+):\s+(.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, component, rest = match.groups()
            
            # CRÍTICO: Dividir SOLO en el PRIMER " - " para que el resto quede intacto
            if ' - ' in rest:
                parts = rest.split(' - ', 1)  # maxsplit=1
                event = parts[0].strip()
                details = parts[1].strip()
            else:
                event = rest.strip()
                details = ''
            
            return {
                'timestamp': datetime.fromisoformat(timestamp_str),
                'component': component,
                'event': event,
                'details': details,
                'format': 'mobile'
            }
        
        # Líneas sin componente
        pattern_generic = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]\s+(.*)'
        match = re.match(pattern_generic, line)
        
        if match:
            timestamp_str, message = match.groups()
            component = 'GENERIC'
            if 'resizing' in message.lower():
                component = 'PREPROCESSING'
            elif 'running inference' in message.lower():
                component = 'INFERENCE'
            elif 'depth' in message.lower():
                component = 'DEPTH'
            
            return {
                'timestamp': datetime.fromisoformat(timestamp_str),
                'component': component,
                'event': message.strip(),
                'details': '',
                'format': 'mobile'
            }
        
        return None
    
    def parse_esp_line(self, line: str, base_time: datetime) -> Optional[Dict]:
        """Parsea línea formato ESP: T+HH:MM:SS.mmm | COMPONENT | message"""
        pattern = r'T\+(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+\|\s+(\w+)\s+\|\s+(.*)'
        match = re.match(pattern, line)
        
        if match:
            hours, minutes, seconds, millis, component, message = match.groups()
            delta = timedelta(
                hours=int(hours),
                minutes=int(minutes),
                seconds=int(seconds),
                milliseconds=int(millis)
            )
            timestamp = base_time + delta
            
            return {
                'timestamp': timestamp,
                'component': component,
                'event': 'capture',
                'details': message.strip(),
                'format': 'esp'
            }
        return None
    
    def calculate_duration_ms(self, start_time: datetime, end_time: datetime) -> float:
        """Calcula duración en milisegundos"""
        delta = end_time - start_time
        return round(delta.total_seconds() * 1000, 2)
    
    def ms_to_hhmmss(self, milliseconds: float) -> str:
        """Convierte milisegundos a formato HH:MM:SS.mmm"""
        if milliseconds is None or pd.isna(milliseconds):
            return None
        
        try:
            milliseconds = float(milliseconds)
            total_seconds = milliseconds / 1000
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            ms = int((milliseconds % 1000))
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
        except (ValueError, TypeError):
            return None
    
    def parse_file_name(self, filename: str) -> Dict:
        """Extrae información del nombre del archivo"""
        parts = filename.replace('.log', '').replace('.txt', '').split(' - ')
        modo = parts[0].strip() if parts else 'UNKNOWN'
        escenarios = ', '.join(parts[1:]) if len(parts) > 1 else ''
        
        return {'modo': modo, 'escenarios': escenarios}
    
    def process_mobile_file(self, filepath: Path, parsed_lines: List[Dict]) -> List[Dict]:
        """Procesa archivo formato MOBILE"""
        records = []
        file_info = self.parse_file_name(filepath.name)
        
        if not parsed_lines:
            return records
        
        cycle_start_time = None
        
        for i, current in enumerate(parsed_lines):
            # Detectar inicio de ciclo
            is_capture_start = (
                current['component'] == 'ESP32' and 
                'capture' in current['event'] and 
                ('starting' in current['details'] or 'starting' in current['event'])
            )
            
            if is_capture_start:
                cycle_start_time = current['timestamp']
            
            # Calcular t_total_ms
            t_total = None
            if i > 0:
                t_total = self.calculate_duration_ms(parsed_lines[i-1]['timestamp'], current['timestamp'])
            
            # Detectar fin de ciclo
            is_cycle_end = (
                current['component'] == 'TTS' and 
                'converted output to speech' in current['event']
            )
            
            # Calcular latencia
            latencia = None
            if is_cycle_end and cycle_start_time is not None:
                latencia = self.calculate_duration_ms(cycle_start_time, current['timestamp'])
                cycle_start_time = None
            
            # EXTRACCIÓN DE DATOS
            objeto_predicho = None
            confianza = None
            
            # SCENE
            if current['component'] == 'SCENE' and 'result' in current['event']:
                match_label = re.search(r'(\w+)\s+with\s+confidence', current['details'])
                if match_label:
                    label = match_label.group(1)
                    objeto_predicho = label if label.lower() != 'unknown' else 'unknown'
                
                match_conf = re.search(r'confidence\s+([\d.]+)', current['details'])
                if match_conf:
                    confianza = float(match_conf.group(1))
            
            # OCR - EXTRACCIÓN CORRECTA DEL TEXTO COMPLETO
            if current['component'] == 'OCR' and 'finished' in current['event']:
                # details contiene: "result: \"texto completo del OCR\""
                if current['details'].startswith('result:'):
                    # Remover "result: " (7 caracteres)
                    ocr_full = current['details'][7:].strip()
                    # Remover comillas externas
                    if ocr_full.startswith('"') and ocr_full.endswith('"'):
                        ocr_full = ocr_full[1:-1]
                    # Limpiar saltos de línea y espacios extras para CSV
                    ocr_full = ' '.join(ocr_full.split())
                    # TODO el texto OCR va a objeto_predicho
                    objeto_predicho = ocr_full

            # Depth
            if current['component'] == 'DEPTH':
                print(f"Depth details: {current['details']}")
                # done. minDistance=171.0 cm (1.710m), globalMin=1.3098082542419434 globalMax=2.3171684741973877
                match_label = re.search(r'minDistance=(\d+(?:\.\d+)?\s*cm)', current['details'])
                if match_label:
                    label = match_label.group(1)
                    print(label)
                    objeto_predicho = label 

            # NOTAS: Siempre contiene la descripción completa del evento
            notas = ''
            if current.get('notas'):
                notas = current['notas']
            elif current['details']:
                notas = current['details']
            elif current['event']:
                notas = current['event']
            
            # Crear registro
            record = {
                'id_prueba': self.id_counter,
                'timestamp': current['timestamp'].isoformat(),
                'modo': file_info['modo'],
                'tipo_evento': current['component'],
                'escenario': file_info['escenarios'],
                'distancia_m': None,
                'iluminacion': None,
                'objeto_verdad': None,
                'objeto_predicho': objeto_predicho,
                'confianza': confianza,
                't_total_ms': t_total,
                'latencia': latencia,
                'acierto': None,
                'notas': notas
            }
            
            records.append(record)
            self.id_counter += 1
        
        return records
    
    def process_esp_file(self, filepath: Path, parsed_lines: List[Dict]) -> List[Dict]:
        """Procesa archivo formato ESP"""
        records = []
        file_info = self.parse_file_name(filepath.name)
        
        for i, line in enumerate(parsed_lines):
            t_total = None
            if i > 0:
                t_total = self.calculate_duration_ms(parsed_lines[i-1]['timestamp'], line['timestamp'])
            
            record = {
                'id_prueba': self.id_counter,
                'timestamp': line['timestamp'].isoformat(),
                'modo': file_info['modo'],
                'tipo_evento': 'ESP32',
                'escenario': file_info['escenarios'],
                'distancia_m': None,
                'iluminacion': None,
                'objeto_verdad': None,
                'objeto_predicho': None,
                'confianza': None,
                't_total_ms': t_total,
                'latencia': None,
                'acierto': None,
                'notas': line['details']
            }
            
            records.append(record)
            self.id_counter += 1
        
        return records
    
    def process_log_file(self, filepath: Path) -> List[Dict]:
        """Procesa un archivo de log"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        lines = None
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            print(f"    ⚠️ Usando UTF-8 con errores reemplazados")
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        
        if not lines:
            return []
        
        # Detectar formato
        first_line = lines[0].strip()
        is_mobile = first_line.startswith('[') and re.match(r'\[\d{4}-\d{2}-\d{2}T', first_line)
        is_esp = first_line.startswith('T+')
        
        parsed_lines = []
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if is_mobile:
                parsed = self.parse_mobile_line(line)
                
                if parsed:
                    # CRITICAL: Si es OCR finished y details empieza con "result: " y contiene comillas
                    # puede ser multi-línea. Verificar si la comilla está cerrada.
                    if (parsed['component'] == 'OCR' and 
                        'finished' in parsed['event'] and 
                        parsed['details'].startswith('result:')):
                        
                        details = parsed['details']
                        # Contar comillas en details
                        quote_count = details.count('"')
                        
                        # Si hay número impar de comillas, el texto continúa en las siguientes líneas
                        if quote_count % 2 == 1:
                            # Leer líneas adicionales hasta cerrar la comilla
                            j = i + 1
                            while j < len(lines):
                                next_line = lines[j].strip()
                                if next_line:
                                    details += '\n' + next_line
                                    quote_count += next_line.count('"')
                                    # Si ahora hay número par de comillas, terminamos
                                    if quote_count % 2 == 0:
                                        i = j  # Actualizar índice para continuar después
                                        break
                                j += 1
                            
                            parsed['details'] = details
                    
                    parsed['notas'] = ''
                    parsed_lines.append(parsed)
                else:
                    # Línea que no se puede parsear - podría ser continuación
                    if parsed_lines and parsed_lines[-1]['format'] == 'mobile':
                        if parsed_lines[-1]['notas']:
                            parsed_lines[-1]['notas'] += ' ' + line
                        else:
                            parsed_lines[-1]['notas'] = line
                
            elif is_esp:
                parsed = self.parse_esp_line(line, base_time)
                if parsed:
                    parsed['notas'] = ''
                    parsed_lines.append(parsed)
            
            i += 1
        
        if not parsed_lines:
            return []
        
        if is_mobile:
            return self.process_mobile_file(filepath, parsed_lines)
        elif is_esp:
            return self.process_esp_file(filepath, parsed_lines)
        
        return []
    
    def parse_all_logs(self, output_csv: str = "resultados_pruebas.csv"):
        """Procesa todos los archivos de log"""
        all_records = []
        
        log_files = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.txt"))
        
        if not log_files:
            print(f"⚠️ No se encontraron archivos de log en {self.logs_dir}")
            return None
        
        output_path = Path(output_csv)
        if output_path.exists():
            print(f"\n⚠️ El archivo '{output_csv}' ya existe.")
            response = input("¿Deseas sobrescribirlo? (s/n): ").lower()
            if response != 's':
                from datetime import datetime as dt
                timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                output_csv = f"resultados_pruebas_{timestamp}.csv"
                print(f"✅ Se guardará como: {output_csv}")
        
        print(f"\n📁 Procesando {len(log_files)} archivo(s)...")
        
        for log_file in log_files:
            print(f"  ⚙️ Procesando: {log_file.name}")
            try:
                records = self.process_log_file(log_file)
                all_records.extend(records)
                print(f"    ✅ {len(records)} registros extraídos")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not all_records:
            print("\n⚠️ No se generaron registros.")
            return None
        
        df = pd.DataFrame(all_records)
        
        # Ordenar por timestamp
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt').reset_index(drop=True)
        
        # Guardar latencia original
        latencia_original = df['latencia'].copy()
        
        # Recalcular t_total_ms
        df['t_total_ms'] = None
        for i in range(1, len(df)):
            prev_time = df.loc[i-1, 'timestamp_dt']
            curr_time = df.loc[i, 'timestamp_dt']
            duration_ms = (curr_time - prev_time).total_seconds() * 1000
            df.loc[i, 't_total_ms'] = round(duration_ms, 2)
        
        # Restaurar latencia
        df['latencia'] = latencia_original
        df = df.drop('timestamp_dt', axis=1)
        
        # Reasignar IDs
        df['id_prueba'] = range(1, len(df) + 1)
        
        # Convertir tiempos a formato HH:MM:SS.mmm
        if 'latencia' in df.columns:
            df['latencia'] = df['latencia'].apply(lambda x: self.ms_to_hhmmss(x) if pd.notna(x) else None)
        if 't_total_ms' in df.columns:
            df['t_total_ms'] = df['t_total_ms'].apply(lambda x: self.ms_to_hhmmss(x) if pd.notna(x) else None)
        
        # Guardar CSV con quotechar para manejar comillas internas
        df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=1)  # quoting=1 = QUOTE_ALL
        print(f"\n✅ CSV generado: {output_csv}")
        print(f"📊 Total de registros: {len(df)}")
        
        return df


if __name__ == "__main__":
    parser = LogParser(logs_directory="./logs")
    df = parser.parse_all_logs(output_csv="resultados_pruebas.csv")
    
    if df is not None and not df.empty:
        print("\n" + "="*60)
        print("📊 RESUMEN DE DATOS")
        print("="*60)
        
        print(f"\n📋 Eventos por componente:")
        print(df['tipo_evento'].value_counts())
        
        print(f"\n🔧 Modos de ejecución:")
        print(df['modo'].value_counts())
        
        latencia_count = df['latencia'].notna().sum()
        if latencia_count > 0:
            print(f"\n⏱️ Ciclos completos (con latencia): {latencia_count}")
        
        t_total_count = df['t_total_ms'].notna().sum()
        if t_total_count > 0:
            print(f"\n⚙️ Eventos con tiempo medido: {t_total_count}")
        
        predictions = df[df['objeto_predicho'].notna()]
        if not predictions.empty:
            print(f"\n🎯 Objetos detectados:")
            print(predictions['objeto_predicho'].value_counts())
        
        print(f"\n📄 Muestra:")
        print(df.head(10).to_string())