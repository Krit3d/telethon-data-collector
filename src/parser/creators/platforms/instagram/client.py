import asyncio
import logging
from typing import Any

from aiohttp import ClientResponseError

logger = logging.getLogger(__name__)


async def fetch_instagram_profile(
    client: Any, handle: str
) -> dict[str, Any] | None:
    try:
        response = await client.get(
            endpoint="/v1/instagram/profile",
            params={"handle": handle},
        )
        data = response.get("data")
        if not data:
            logger.error(
                "Missing 'data' in API response for Instagram handle %s",
                handle,
            )
            return None

        user = data.get("user") or data
        if not user:
            logger.error(
                "Missing user data for Instagram handle %s", handle
            )
            return None

        return user

    except ClientResponseError as e:
        if e.status == 404:
            logger.warning(
                "Instagram profile %s not found (404). Marking as rejected.",
                handle,
            )
        else:
            logger.error(
                "Instagram API request failed for %s with HTTP %d: %s",
                handle,
                e.status,
                e,
            )
        return None
    except Exception as e:
        logger.error(
            "Unexpected error fetching Instagram profile %s: %s",
            handle,
            e,
            exc_info=True,
        )
        return None


async def fetch_video_transcript(
    client: Any, semaphore: asyncio.Semaphore, post_url: str
) -> str | None:
    logger.debug("Requesting transcript: %s", post_url[:50])

    async with semaphore:
        try:
            response = await client.get(
                endpoint="/v2/instagram/media/transcript",
                params={"url": post_url},
                max_retries=2,
            )
            transcripts = response.get("transcripts")
            if isinstance(transcripts, list) and len(transcripts) > 0:
                first_item = transcripts[0]

                if first_item is None:
                    logger.debug("No transcript: %s", post_url[:50])
                    return None

                transcript_text: str | None = None

                if isinstance(first_item, str) and first_item:
                    transcript_text = first_item
                elif isinstance(first_item, dict):
                    text_value = first_item.get("text")
                    if isinstance(text_value, str) and text_value:
                        transcript_text = text_value

                if transcript_text:
                    if "please provide the video or audio file you would like me to transcribe" in transcript_text.lower():
                        logger.warning(
                            "Transcript API returned placeholder error text for post: %s",
                            post_url[:50],
                        )
                        return None
                    logger.debug("Got transcript: %s", post_url[:50])
                    return transcript_text

            logger.debug("No transcript: %s", post_url[:50])
            return None

        except ClientResponseError as e:
            if e.status >= 500:
                logger.warning(
                    "Transcript failed for %s (HTTP %d: video might exceed 2 minutes, contain no speech, or be unavailable on Scrape Creators side).",
                    post_url,
                    e.status,
                )
            else:
                logger.warning(
                    "Transcript request error for %s (HTTP %d): %s",
                    post_url,
                    e.status,
                    e,
                )
            return None

        except Exception as e:
            logger.warning(
                "Transcript failed for %s: %s", post_url, e
            )
            return None
