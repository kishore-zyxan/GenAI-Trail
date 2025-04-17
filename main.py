from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Dict
import os, json, re

from extractor import extract_text
from llm_client import analyze_with_llm
from database import connect_mysql
from crud import insert_into_mysql, fetch_uploaded_files, delete_files_by_ids, update_or_insert_file
from utils import flatten_json, compute_diff
from models import FileDeleteRequest

app = FastAPI(title="Universal Document Reader")

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    responses = []

    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        content = await file.read()

        try:
            text = extract_text(file_ext, content)
            if not text:
                raise HTTPException(status_code=400, detail="Text extraction failed.")

            # LLM Analysis
            llm_output = analyze_with_llm(text)
            match = re.search(r"\{(?:[^{}]*|\{(?:[^{}]*|\{.*})*})*}" , llm_output, re.DOTALL)
            if not match:
                raise HTTPException(status_code=500, detail="No JSON object found in LLM output.")
            json_str = match.group(0).strip()
            data = json.loads(json_str)
            flat_data = flatten_json(data)

            # Store in DB
            insert_into_mysql(file.filename, flat_data)

            responses.append({
                "filename": file.filename,
                "extracted_data": flat_data
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return JSONResponse(content={"results": responses})

@app.get("/files/")
def get_uploaded_files():
    return fetch_uploaded_files()

@app.delete("/delete/")
def delete_files(req: FileDeleteRequest):
    delete_files_by_ids(req.file_ids)
    return {"message": "Selected files deleted successfully"}

@app.get("/file/{file_id}")
def get_file_by_id(file_id: int):
    try:
        conn = connect_mysql()
        cursor = conn.cursor()
        cursor.execute("SELECT file_name, json_data, update_count, diff_data FROM documents WHERE id = %s", (file_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="File not found")

        file_name, json_data, update_count, diff_data = result
        return {
            "file_id": file_id,
            "file_name": file_name,
            "json_data": json.loads(json_data),
            "update_count": update_count,
            "diff_data": json.loads(diff_data) if diff_data else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/file/{file_id}")
def update_file_json(file_id: int, new_data: Dict = Body(...)):
    try:
        conn = connect_mysql()
        cursor = conn.cursor()
        cursor.execute("SELECT json_data, update_count FROM documents WHERE id = %s", (file_id,))
        result = cursor.fetchone()

        if result:
            old_json_str, update_count = result
            old_json = json.loads(old_json_str)
            diff = compute_diff(old_json, new_data)

            update_count += 1
            cursor.execute("""
                UPDATE documents
                SET json_data = %s, update_count = %s, diff_data = %s
                WHERE id = %s
            """, (json.dumps(new_data), update_count, json.dumps(diff), file_id))
        else:
            # New file insert as PUT if not exists
            cursor.execute("""
                INSERT INTO documents (file_name, json_data, update_count, diff_data)
                VALUES (%s, %s, %s, %s)
            """, (f"file_{file_id}", json.dumps(new_data), 0, json.dumps({})))

        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "JSON data updated successfully", "updated_data": new_data, "diff": diff if result else {}}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
