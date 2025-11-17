"""
File system operations API routes
"""
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ...core.logging import get_logger
from ...models.schemas import FolderInfoResponse, ImagePreviewResponse
from ...services.file_service import get_file_service

logger = get_logger(__name__)
router = APIRouter()


@router.get("/folder/{path:path}", response_model=List[FolderInfoResponse])
async def get_folder_info(path: str):
    """
    Get folder contents

    Args:
        path: URL-encoded folder path
    """
    try:
        # Decode URL path
        decoded_path = path.replace("%20", " ")
        folder_path = Path(decoded_path)

        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        file_service = get_file_service()
        contents = await file_service.get_folder_contents(folder_path)

        return [
            FolderInfoResponse(
                name=item.name,
                path=str(item.path),
                type="directory" if item.path.is_dir() else "file",
                size=item.size,
                modified=item.modified,
                children_count=item.children_count
            )
            for item in contents
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get folder info for {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/preview/{path:path}")
async def get_image_preview(path: str, size: int = Query(150, ge=50, le=1000)):
    """
    Get image preview

    Args:
        path: URL-encoded image path
        size: Preview size in pixels
    """
    try:
        # Decode URL path
        decoded_path = path.replace("%20", " ")
        image_path = Path(decoded_path)

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        file_service = get_file_service()
        preview = file_service.get_image_preview(image_path, size)

        if preview is None:
            raise HTTPException(status_code=500, detail="Failed to generate preview")

        return ImagePreviewResponse(
            content=preview,
            content_type="image/jpeg"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image preview for {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/folder")
async def create_folder(
    path: str = Query(..., description="Parent directory path"),
    name: str = Query(..., description="New folder name")
):
    """
    Create new folder

    Args:
        path: Parent directory path
        name: New folder name
    """
    try:
        parent_path = Path(path)
        if not parent_path.exists() or not parent_path.is_dir():
            raise HTTPException(status_code=400, detail="Invalid parent directory")

        file_service = get_file_service()
        new_folder = await file_service.create_folder(parent_path, name)

        return {
            "success": True,
            "path": str(new_folder)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create folder {name} in {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/item/{path:path}")
async def delete_item(path: str):
    """
    Delete file or folder

    Args:
        path: Path to delete
    """
    try:
        # Decode URL path
        decoded_path = path.replace("%20", " ")
        item_path = Path(decoded_path)

        if not item_path.exists():
            raise HTTPException(status_code=404, detail="Item not found")

        file_service = get_file_service()
        await file_service.delete_item(item_path)

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete item {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/move")
async def move_item(
    srcPath: str = Query(..., description="Source path"),
    destPath: str = Query(..., description="Destination path")
):
    """
    Move file or folder

    Args:
        srcPath: Source path
        destPath: Destination path
    """
    try:
        source_path = Path(srcPath)
        dest_path = Path(destPath)

        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source item not found")

        file_service = get_file_service()
        await file_service.move_item(source_path, dest_path)

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to move item from {srcPath} to {destPath}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
